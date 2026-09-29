#!/usr/bin/env python3
"""Step 4b: train and evaluate the edit verifier.

The verifier sees a stretch of Whisper text with one spot marked, the words
Whisper wrote there, and a proposed replacement, and answers YES (apply it)
or NO (keep Whisper's words). Its score is P(YES) / (P(YES) + P(NO)) from the
first answer token, so the acceptance threshold can be tuned for precision.

Pipeline it's evaluated in (and meant for):
  Whisper -> rules.py -> the rewrite model's output turned into candidate
  edits (edits.extract_edits; substitutions only, no style-only or
  filler-only changes, <= max_span_words) -> the verifier scores each ->
  candidates above the threshold are applied to the Whisper text.

Evaluation after training:
  * validation and calibration splits (build_verifier_data.py): AUC and
    precision/recall of YES per threshold -- the calibration split is scored
    against hand labels;
  * the test set, with candidates from a rewrite run's predictions.jsonl
    (eval.candidates_file): WER and words fixed/broken per threshold next to
    Whisper, rules only, accepting every candidate, and the oracle (only the
    candidates that help), per slice.

  python3 src/edit/train_verifier.py --config configs/edit/qwen3.5-2b-verifier.yaml
  # re-evaluate a trained verifier on other candidates, no training (results go
  # to a new eval-<candidates run>-<time>/ subfolder of the run, locally and on the Hub):
  python3 src/edit/train_verifier.py --config configs/edit/qwen3.5-2b-verifier.yaml --eval-only \\
      --set eval.candidates_file=<run>/test_eval_best/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

import torch  # noqa: E402
from datasets import Dataset, DatasetDict  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from build_diffs import is_filler_edit  # noqa: E402
from edits import Edit, apply_edits, extract_edits, validate_edits  # noqa: E402
from hub_sync import sync_output_dir  # noqa: E402
from hub_utils import ensure_new_folder, upload_new_folder  # noqa: E402
from normalize import is_style_edit, strip_punctuation  # noqa: E402
from scoring import lenient_positions, score  # noqa: E402
from train_edit import (  # noqa: E402
    chat_prompts, load_edit_config, load_lora_model, load_test_set, make_trainer, prepare_output_dir, tokenize_examples,
)

load_dotenv()

SYSTEM_PROMPT = (
    "You check proposed corrections to speech-recognition transcripts of Persian customer-service phone calls "
    "(Digikala, an online store). You get the transcript around one spot, the words the recognizer wrote there, "
    "and a proposed replacement, and sometimes the names of the people on the call from the store's records.\n"
    "Answer YES only if the recognizer clearly misheard and the replacement is what was actually said -- a word "
    "that doesn't fit the sentence, a misheard name, brand or product. Answer NO if the original words are fine, "
    "if both are acceptable (spelling, half-space, colloquial vs formal, filler words), or if you can't tell. "
    "Output only YES or NO."
)
YES, NO = "YES", "NO"

DEFAULTS = {
    "model_id": "Qwen/Qwen3.5-2B",
    "output_dir": "./outputs/verifier-qwen3.5-2b",
    "max_length": 512,
    "data": {
        # build_verifier_data.py's split files -- locally, or the folder they
        # were uploaded to in hub_repo (hub_folder set: read from the Hub).
        "sources": {"train": "outputs/edit/verifier/train.jsonl",
                    "validation": "outputs/edit/verifier/validation.jsonl",
                    "calibration": "outputs/edit/verifier/calibration.jsonl"},
        "hub_repo": "PedramR/ASR_Post-processing",
        "hub_folder": None,
    },
    "training": {
        "num_train_epochs": 2.0, "max_steps": -1,
        "per_device_train_batch_size": 8, "per_device_eval_batch_size": 8, "gradient_accumulation_steps": 2,
        "learning_rate": 1.0e-4, "warmup_ratio": 0.05, "lr_scheduler_type": "linear",
        "logging_steps": 10, "eval_steps": 50, "save_steps": 50, "save_total_limit": 2, "seed": 42,
    },
    "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
    "hub": {"push_to_hub": False, "repo_id": "PedramR/ASR_Post-processing", "folder": None, "private": True},
    "tensorboard": {"logging_dir": None},
    "eval": {
        "test_dataset_id": "ErfanRou/callcc-test-1k",
        "slices_dataset_id": "PedramR/ASR_Post-processing-eval",
        "corpus_freq_repo": "PedramR/ASR_Post-processing-dataset",
        "entities_dir": "data/output-backup",
        "max_examples": None,
        # the rewrite run whose test predictions provide the candidate edits
        "candidates_repo": "PedramR/ASR_Post-processing",
        "candidates_file": "qwen3.5-2b-100pct-masked-weighted-v2/test_eval_best/predictions.jsonl",
        "context_words": 12,
        "max_span_words": 4,
        "threshold": 0.5,  # the one written to test_eval/metrics.json
        "thresholds": [0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
        "batch_size": 32,
    },
}


# --------------------------------------------------------------------------- format

def render_input(item: dict) -> str:
    """`item`: left / original / proposed / right (+ crm_names), as written by
    build_verifier_data.py and extract_candidates()."""
    names = item.get("crm_names") or []
    head = f"Known names: {'، '.join(names)}\n" if names else ""
    original = item["original"] or "∅"
    return (f"{head}Context: {item['left']} ⟦{original}⟧ {item['right']}\n"
            f"Proposed change: ⟦{original}⟧ → ⟦{item['proposed']}⟧")


# --------------------------------------------------------------------------- data

def load_verifier_data(cfg) -> DatasetDict:
    splits = {}
    for name, path in cfg.data.sources.items():
        if cfg.data.hub_folder:
            path = hf_hub_download(cfg.data.hub_repo, f"{cfg.data.hub_folder.strip('/')}/{Path(path).name}")
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if rows:
            splits[name] = Dataset.from_list(rows)
    print({k: f"{len(v)} ({sum(r['label'] == YES for r in v)} YES)" for k, v in splits.items()})
    return DatasetDict(splits)


# --------------------------------------------------------------------------- scoring

def make_scorer(model, tokenizer, batch_size: int):
    """prompts-free scorer: items -> P(YES) / (P(YES) + P(NO))."""
    from decode_sweep import first_token_probs

    yes_id = tokenizer(YES, add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer(NO, add_special_tokens=False)["input_ids"][0]

    def scorer(items: list[dict]) -> list[float]:
        if not items:
            return []
        model.eval()
        prompts = chat_prompts(tokenizer, [render_input(it) for it in items], SYSTEM_PROMPT)
        probs = first_token_probs(model, tokenizer, prompts, [yes_id, no_id], batch_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [y / (y + n) if y + n > 0 else 0.0 for y, n in probs]

    return scorer


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def decision_report(rows, scores: list[float], thresholds: list[float]) -> dict:
    labels = [r["label"] == YES for r in rows]
    out = {"examples": len(rows), "yes": sum(labels), "auc": auc(scores, labels), "by_threshold": {}}
    for t in thresholds:
        acc = [s >= t for s in scores]
        tp = sum(a and l for a, l in zip(acc, labels))
        out["by_threshold"][str(t)] = {"accepted": sum(acc), "precision": tp / sum(acc) if sum(acc) else None,
                                       "recall": tp / sum(labels) if sum(labels) else None,
                                       "accuracy": sum(a == l for a, l in zip(acc, labels)) / len(rows)}
    return out


# --------------------------------------------------------------------------- candidates from a rewrite run

def extract_candidates(after_rules: str, rewrite_output: str, names: list[str], context_words: int,
                       max_span_words: int) -> list[dict]:
    """The rewrite model's changes to the (rules-fixed) Whisper text, as
    verifier items: substitutions of <= max_span_words words that aren't
    only a spelling/dialect variant or filler words."""
    words = after_rules.split()
    out = []
    for e in extract_edits(after_rules, strip_punctuation(rewrite_output)):
        if (e.kind != "substitute" or max(len(e.original), len(e.replacement)) > max_span_words
                or is_style_edit(e) or is_filler_edit(e)):
            continue
        out.append({"start": e.start, "end": e.end, "original": " ".join(e.original), "proposed": " ".join(e.replacement),
                    "left": " ".join(words[max(0, e.start - context_words):e.start]),
                    "right": " ".join(words[e.end:e.end + context_words]), "crm_names": names})
    return out


def candidate_effect(after_rules: str, reference: str, c: dict) -> int:
    """Lenient words fixed minus broken by applying candidate `c` alone."""
    target, already, still, _ = lenient_positions(after_rules, after_rules, reference)
    edit = Edit(c["start"], c["end"], c["original"].split(), c["proposed"].split())
    t2, a2, s2, _ = lenient_positions(after_rules, apply_edits(after_rules, [edit]), reference)
    return (len(t2 & s2) - len(target & still)) - (len(already & still) - len(a2 & s2))


def apply_accepted(after_rules: str, cands: list[dict], keep) -> str:
    edits = [Edit(c["start"], c["end"], c["original"].split(), c["proposed"].split()) for c in cands if keep(c)]
    valid, _ = validate_edits(after_rules.split(), sorted(edits, key=lambda e: e.start))
    return apply_edits(after_rules, valid)


def evaluate_candidates(cfg, scorer, out_dir: Path) -> dict:
    """Test-set evaluation with the rewrite run's changes as candidates (see
    module docstring). Rows the rewrite model stopped early on
    (premature_stop) contribute no candidates: their "edits" are a
    truncation, not corrections."""
    e = cfg.eval
    ts = load_test_set(cfg)
    by_text = {w: i for i, w in enumerate(ts.whisper)}
    preds = [json.loads(line) for line in open(hf_hub_download(e.candidates_repo, e.candidates_file), encoding="utf-8")]
    rewrite = {}
    skipped = Counter()
    for p in preds:
        i = by_text.get(p["input"])
        if i is None:
            skipped["not in the test set"] += 1
        elif p.get("premature_stop"):
            skipped["premature_stop"] += 1
        else:
            rewrite[i] = p["output"]
    cands = [extract_candidates(ts.after_rules[i], rewrite[i], ts.names[i], e.context_words, e.max_span_words)
             if i in rewrite else [] for i in range(len(ts.rows))]
    flat = [(i, c) for i, cs in enumerate(cands) for c in cs]
    print(f"Candidates: {len(flat)} from {len(rewrite)} rewrite outputs (skipped rows: {dict(skipped)})")
    for (i, c), s in zip(flat, scorer([c for _, c in flat])):
        c["p_yes"] = s
    for i, c in flat:
        c["effect"] = candidate_effect(ts.after_rules[i], ts.refs[i], c)

    methods = {"rules only": lambda c: False, "accept all": lambda c: True, "oracle": lambda c: c["effect"] > 0}
    methods.update({f"verifier@{t}": (lambda c, t=t: c["p_yes"] >= t) for t in e.thresholds})
    if f"verifier@{e.threshold}" not in methods:
        methods[f"verifier@{e.threshold}"] = lambda c: c["p_yes"] >= e.threshold
    hyps = {m: [apply_accepted(ts.after_rules[i], cands[i], keep) for i in range(len(ts.rows))] for m, keep in methods.items()}

    table = ["| slice | method | WER | fixed | broken | net | broken/100 | entity fix | accepted |",
             "|---|---|---|---|---|---|---|---|---|"]
    results = {}
    for name, idx in ts.slices.items():
        pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
        results[name] = {}
        for m, keep in [("whisper", None)] + list(methods.items()):
            h = ts.whisper if m == "whisper" else hyps[m]
            s = score(pick(ts.whisper), pick(h), pick(ts.refs), ts.freq, pick(ts.spans))
            s["accepted"] = 0 if keep is None else sum(keep(c) for i in idx for c in cands[i])
            results[name][m] = s
            table.append(f"| {name} | {m} | {s['wer']:.4f} | {s['fixed_lenient']} | {s['broken_lenient']} | "
                         f"{s['net_fixed_lenient']} | {s['broken_per_100_correct']:.2f} | "
                         f"{(s.get('entity_fix_rate') or 0):.3f} | {s['accepted']} |")
        slice_dir = out_dir / ("test_eval" if name == "main" else f"test_eval_{name}")
        slice_dir.mkdir(parents=True, exist_ok=True)
        (slice_dir / "metrics.json").write_text(json.dumps(results[name][f"verifier@{e.threshold}"], indent=2))
        (slice_dir / "baselines.json").write_text(json.dumps(
            {m: v for m, v in results[name].items() if m != f"verifier@{e.threshold}"}, indent=2))
        with open(slice_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps({"call_id": ts.rows[i]["call_id"], "input": ts.whisper[i], "after_rules": ts.after_rules[i],
                                    "rewrite_output": rewrite.get(i), "output": hyps[f"verifier@{e.threshold}"][i],
                                    "reference": ts.refs[i], "candidates": cands[i]}, ensure_ascii=False) + "\n")
    header = [f"Candidates: {e.candidates_repo}/{e.candidates_file} ({len(flat)} candidates; skipped rows {dict(skipped)})\n"]
    (out_dir / "test_eval" / "comparison.md").write_text("\n".join(header + table) + "\n", encoding="utf-8")
    print("\n".join(header + table))
    return results["main"][f"verifier@{e.threshold}"]


def evaluate_splits(cfg, scorer, data: DatasetDict, out_dir: Path) -> dict:
    report = {}
    for name in ("validation", "calibration"):
        if name not in data:
            continue
        rows = list(data[name])
        scores = scorer(rows)
        report[name] = decision_report(rows, scores, cfg.eval.thresholds)
        with open(out_dir / f"{name}_predictions.jsonl", "w", encoding="utf-8") as f:
            for r, s in zip(rows, scores):
                f.write(json.dumps({**r, "p_yes": s}, ensure_ascii=False) + "\n")
    (out_dir / "verifier_scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- main

def load_trained(cfg):
    """Base model + this run's adapter (local output_dir, else its Hub folder)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16, device_map="auto")
    local = Path(cfg.output_dir) / "adapter_config.json"
    model = (PeftModel.from_pretrained(base, cfg.output_dir) if local.exists()
             else PeftModel.from_pretrained(base, cfg.hub_repo_id, subfolder=cfg.hub_repo_folder or Path(cfg.output_dir).name))
    return model.eval(), tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--eval-only", action="store_true", help="Evaluate an already-trained verifier (no training).")
    args = p.parse_args()
    cfg = load_edit_config(args.config, args.overrides, defaults=DEFAULTS)
    run_folder = cfg.hub_repo_folder or Path(cfg.output_dir).name
    if args.eval_only:
        # A new subfolder of the run, named after the candidates and the time:
        # the run's own files are never overwritten.
        eval_name = f"eval-{cfg.eval.candidates_file.split('/')[0]}-{time.strftime('%Y%m%d-%H%M')}"
        if cfg.hub.push_to_hub:
            ensure_new_folder(cfg.hub_repo_id, f"{run_folder}/{eval_name}")
        out_dir = Path(cfg.output_dir) / eval_name
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if cfg.hub.push_to_hub:
            ensure_new_folder(cfg.hub_repo_id, run_folder)  # a new run never lands in an existing folder
        out_dir = prepare_output_dir(cfg)
    data = load_verifier_data(cfg)

    if args.eval_only:
        model, tokenizer = load_trained(cfg)
    else:
        model, tokenizer = load_lora_model(cfg)
        tokenized = tokenize_examples(tokenizer, DatasetDict({k: v for k, v in data.items() if k in ("train", "validation")}),
                                      lambda r: (render_input(r), r["label"]), SYSTEM_PROMPT, cfg.max_length)
        trainer = make_trainer(cfg, model, tokenizer, tokenized)
        trainer.train()
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

    scorer = make_scorer(model, tokenizer, cfg.eval.batch_size)
    evaluate_splits(cfg, scorer, data, out_dir)
    if cfg.eval.candidates_file:
        final = evaluate_candidates(cfg, scorer, out_dir)
        print(f"Final (main test set, threshold {cfg.eval.threshold}): wer={final['wer']:.4f}, "
              f"net_fixed_lenient={final['net_fixed_lenient']}, broken_per_100_correct={final['broken_per_100_correct']:.2f}")
    if cfg.hub.push_to_hub and args.eval_only:
        print("Uploaded to", upload_new_folder(cfg.hub_repo_id, out_dir, f"{run_folder}/{out_dir.name}", "verifier eval"))
    elif cfg.hub.push_to_hub:
        sync_output_dir(cfg, commit_message="final")


if __name__ == "__main__":
    main()
