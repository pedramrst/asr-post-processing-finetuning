#!/usr/bin/env python3
"""Step 3 of the edit-based method: LoRA SFT of an edit model, and its
evaluation on full test transcripts.

The model sees one Whisper window with numbered words and outputs a short
edit list (edit_format.py); code applies the edits, so every other word stays
Whisper's own. Training data is generate_targets.py's train.jsonl files (or a
Hub dataset of them, see publish_edit_dataset.py).

Evaluation (end of training, and optionally before it) runs the full pipeline
on the test set exactly as it would run in production: rules.py first, then
overlapping windows (windowing.py) through the model, edits merged back into
the whole transcript, scored with scoring.py next to two baselines -- Whisper
untouched and rules only -- on the main test set and the entity/typo slices.

Same machinery as src/train.py where it applies (LoRA via peft, prompt
masking via data.build_example, PadCollator, TensorBoard, hub_sync.py's
folder-per-run Hub layout), but its own config schema, since the rewrite
pipeline's options (low-signal masking, short-target EOS weighting, ...)
assume a full-transcript target. run_sweep.py can drive it:

  python3 src/edit/train_edit.py --config configs/edit/qwen3.5-2b-edit.yaml
  python3 src/edit/train_edit.py --config configs/edit/smoke.yaml --set training.max_steps=5
  python3 src/run_sweep.py --sweep configs/edit/sweep.yaml --train_script src/edit/train_edit.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import Dataset, DatasetDict, load_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

from corpus import load_corpus_freq  # noqa: E402
from data import PadCollator, build_example  # noqa: E402
from edit_format import parse_edits, render_input, render_target, system_prompt  # noqa: E402
from edits import apply_edits  # noqa: E402
from evaluate import _load_hub_columns_pruned, generate_batch  # noqa: E402
from grounding import crm_candidates, load_gemini_entities, reference_entity_spans  # noqa: E402
from hub_sync import SyncToHubCallback, sync_output_dir  # noqa: E402
from rules import rule_edits  # noqa: E402
from scoring import score  # noqa: E402
from windowing import build_window_prompts, merge_outputs  # noqa: E402

load_dotenv()

# Every key a config may set, with its default. Unknown keys are an error
# (a typo would otherwise silently fall back to the default).
DEFAULTS = {
    "model_id": "Qwen/Qwen3.5-2B",
    "output_dir": "./outputs/edit-qwen3.5-2b",
    "max_length": 1024,
    "data": {
        # Local train.jsonl files from generate_targets.py, or one Hub dataset id
        # with train/validation splits (publish_edit_dataset.py).
        "sources": ["outputs/edit/data/train.jsonl", "outputs/edit/data_v2/train.jsonl"],
        "min_window_words": 10,
        "target_includes_original": True,
        "train_fraction": None,
    },
    "training": {
        "num_train_epochs": 3.0,
        "max_steps": -1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1.0e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "linear",
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 2,
        "seed": 42,
    },
    "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
    "hub": {"push_to_hub": False, "repo_id": "PedramR/ASR_Post-processing", "folder": None, "private": True},
    "tensorboard": {"logging_dir": None},  # null -> <output_dir>/tb; run_sweep.py sets it per job
    "eval": {
        "test_dataset_id": "ErfanRou/callcc-test-1k",
        "slices_dataset_id": "PedramR/ASR_Post-processing-eval",
        "corpus_freq_repo": "PedramR/ASR_Post-processing-dataset",
        "entities_dir": "data/output-backup",
        "window_words": 50,
        "stride": 25,
        "batch_size": 32,
        "max_new_tokens": 128,
        "max_examples": None,
        "baseline": False,
        "validation_generation": True,
    },
}


# --------------------------------------------------------------------------- config

def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k not in base:
            raise KeyError(f"unknown config key {path + k!r} (known: {sorted(base)})")
        out[k] = _merge(base[k], v, f"{path}{k}.") if isinstance(base[k], dict) and isinstance(v, dict) else v
    return out


def load_edit_config(path: str, overrides: list[str]) -> SimpleNamespace:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:  # --set section.key=value (value parsed as YAML)
        key, _, value = item.partition("=")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    cfg = _merge(DEFAULTS, raw)
    ns = SimpleNamespace(raw=cfg, **{k: v for k, v in cfg.items() if not isinstance(v, dict)})
    for section, values in cfg.items():
        if isinstance(values, dict):
            setattr(ns, section, SimpleNamespace(**values))
    # the attribute names hub_sync.py reads
    ns.hub_repo_id, ns.hub_repo_folder, ns.hub_private = cfg["hub"]["repo_id"], cfg["hub"]["folder"], cfg["hub"]["private"]
    return ns


# --------------------------------------------------------------------------- data

def load_windows(cfg) -> DatasetDict:
    """train/validation windows, keeping only substitution-only windows of at
    least min_window_words words (insertions/deletions are out of scope for
    v1; very short windows add little)."""
    sources = cfg.data.sources if isinstance(cfg.data.sources, list) else [cfg.data.sources]
    rows = []
    if len(sources) == 1 and not Path(sources[0]).exists():
        ds = load_dataset(sources[0])
        rows = [r for split in ds.values() for r in split]
    else:
        for path in sources:
            rows += [json.loads(line) for line in open(path, encoding="utf-8")]
    kept, dropped = {"train": [], "validation": []}, Counter()
    for r in rows:
        if len(r["source"].split()) < cfg.data.min_window_words:
            dropped["shorter than min_window_words"] += 1
        elif any(not e["original"] or not e["replacement"] for e in r["edits"]):
            dropped["has an insertion/deletion"] += 1
        else:
            kept[r["split"]].append(r)
    if cfg.data.train_fraction:
        n = round(len(kept["train"]) * cfg.data.train_fraction)
        kept["train"] = Dataset.from_list(kept["train"]).shuffle(seed=cfg.training.seed).select(range(n)).to_list()
    print(f"Windows: {len(kept['train'])} train, {len(kept['validation'])} validation; dropped {dict(dropped)}")
    return DatasetDict({k: Dataset.from_list(v) for k, v in kept.items() if v})


def example_texts(row: dict, include_original: bool) -> tuple[str, str]:
    return (render_input(row["source"].split(), row.get("crm_names") or []),
            render_target(row["edits"], include_original))


# --------------------------------------------------------------------------- evaluation

def chat_prompts(tokenizer, users: list[str], sys_prompt: str) -> list[str]:
    return [tokenizer.apply_chat_template([{"role": "system", "content": sys_prompt}, {"role": "user", "content": u}],
                                          tokenize=False, add_generation_prompt=True) for u in users]


def evaluate_on_test(model, tokenizer, cfg, out_dir: Path, tag: str = "") -> dict:
    """Full-transcript evaluation (see module docstring). Writes
    test_eval{tag}/metrics.json (model on the main test set, with "wer" at the
    top level for run_sweep.py), baselines.json (whisper / rules on the same
    rows), predictions.jsonl, comparison.md, and the entity/typo slices'
    metrics.json/baselines.json under test_eval_entity{tag}/ and
    test_eval_typo{tag}/."""
    e = cfg.eval
    test = _load_hub_columns_pruned(e.test_dataset_id, ["call_id", "text_whisper", "text", "crm_metadata"], "test")
    rows = [r for r in test if r["text_whisper"] and r["text"]]
    if e.max_examples:
        rows = rows[:e.max_examples]
    slices = {"main": set(range(len(rows)))}
    if e.slices_dataset_id:
        ev = _load_hub_columns_pruned(e.slices_dataset_id, ["text_whisper", "row_type"], "test")
        by_text = {r["text_whisper"]: i for i, r in enumerate(rows)}
        for r in ev:
            if r["text_whisper"] in by_text:
                slices.setdefault(r["row_type"], set()).add(by_text[r["text_whisper"]])

    freq = load_corpus_freq(e.corpus_freq_repo)
    gemini = load_gemini_entities(e.entities_dir) if e.entities_dir and Path(e.entities_dir).exists() else {}
    names = [sorted(crm_candidates(r["crm_metadata"])) for r in rows]
    whisper = [r["text_whisper"] for r in rows]
    after_rules = [apply_edits(w, rule_edits(w, freq, protected=set(n))) for w, n in zip(whisper, names)]
    refs = [r["text"] for r in rows]
    spans = [reference_entity_spans(ref, set(n), gemini.get(r["call_id"], [])) for ref, n, r in zip(refs, names, rows)]

    index, users = build_window_prompts(after_rules, names, e.window_words, e.stride)
    print(f"Evaluating on {len(rows)} transcripts ({len(users)} windows)...")
    was_training = model.training
    model.eval()
    outputs = generate_batch(model, tokenizer, chat_prompts(tokenizer, users, system_prompt(cfg.data.target_includes_original)),
                             e.max_new_tokens, e.batch_size)
    model.train(was_training)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = merge_outputs(after_rules, index, outputs)
    corrected = [r.corrected for r in results]

    run_dir = out_dir / f"test_eval{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r, w, ar, res, ref in zip(rows, whisper, after_rules, results, refs):
            f.write(json.dumps({"input": w, "after_rules": ar, "output": res.corrected, "reference": ref,
                                "edits": [{"start": x.start, "end": x.end, "original": " ".join(x.original),
                                           "replacement": " ".join(x.replacement)} for x in res.edits],
                                "rejected_lines": res.rejected_lines, "windows": res.windows}, ensure_ascii=False) + "\n")

    table, all_metrics = [], {}
    for name, idx in slices.items():
        idx = sorted(idx)
        pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
        per_method = {method: score(pick(whisper), pick(hyps), pick(refs), freq, pick(spans))
                      for method, hyps in (("whisper", whisper), ("rules", after_rules), ("model", corrected))}
        all_metrics[name] = per_method
        m = dict(per_method["model"])
        m["rejected_lines_per_transcript"] = sum(len(results[i].rejected_lines) for i in idx) / max(len(idx), 1)
        m["edits_per_transcript"] = sum(len(results[i].edits) for i in idx) / max(len(idx), 1)
        slice_dir = run_dir if name == "main" else out_dir / f"test_eval_{name}{tag}"
        slice_dir.mkdir(parents=True, exist_ok=True)
        (slice_dir / "metrics.json").write_text(json.dumps(m, indent=2))  # flat: run_sweep.py prints it
        (slice_dir / "baselines.json").write_text(json.dumps(
            {k: v for k, v in per_method.items() if k != "model"}, indent=2))
        for method, v in per_method.items():
            table.append(f"| {name} | {method} | {v['wer']:.4f} | {v['fixed_lenient']} | {v['broken_lenient']} | "
                         f"{v['net_fixed_lenient']} | {v['broken_per_100_correct']:.2f} | "
                         f"{(v.get('entity_fix_rate') or 0):.3f} | {v['rows_changed']:.3f} |")
    header = ["| slice | method | WER | fixed | broken | net | broken/100 | entity fix | rows changed |",
              "|---|---|---|---|---|---|---|---|---|"]
    (run_dir / "comparison.md").write_text("\n".join(header + table) + "\n", encoding="utf-8")
    print("\n".join(header + table))
    return all_metrics["main"]["model"]


def evaluate_validation_edits(model, tokenizer, cfg, windows: Dataset, out_dir: Path) -> dict:
    """Edit-level accuracy on the held-out validation windows: exact-match
    precision/recall of the model's edits against the verified targets, and
    how often its output lines were rejected as malformed."""
    users = [render_input(r["source"].split(), r.get("crm_names") or []) for r in windows]
    model.eval()
    outputs = generate_batch(model, tokenizer, chat_prompts(tokenizer, users, system_prompt(cfg.data.target_includes_original)),
                             cfg.eval.max_new_tokens, cfg.eval.batch_size)
    tp = n_pred = n_gold = rejected = exact = 0
    for r, out in zip(windows, outputs):
        parsed = parse_edits(out, r["source"].split())
        pred = {(e.start, e.end, " ".join(e.replacement)) for e in parsed.edits}
        gold = {(e["start"], e["end"], e["replacement"]) for e in r["edits"]}
        tp += len(pred & gold)
        n_pred += len(pred)
        n_gold += len(gold)
        rejected += len(parsed.rejected)
        exact += pred == gold
    m = {"windows": len(windows), "edit_precision": tp / n_pred if n_pred else None,
         "edit_recall": tp / n_gold if n_gold else None, "window_exact_match": exact / len(windows),
         "rejected_lines": rejected, "predicted_edits": n_pred, "gold_edits": n_gold}
    (out_dir / "validation_edits.json").write_text(json.dumps(m, indent=2))
    print("Validation edits:", m)
    return m


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set training.max_steps=5 (repeatable).")
    args = p.parse_args()
    cfg = load_edit_config(args.config, args.overrides)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False, allow_unicode=True))
    os.environ["TENSORBOARD_LOGGING_DIR"] = cfg.tensorboard.logging_dir or str(out_dir / "tb")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                                             bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()

    windows = load_windows(cfg)
    sys_prompt = system_prompt(cfg.data.target_includes_original)

    def _map(row):
        user, target = example_texts(row, cfg.data.target_includes_original)
        return build_example(tokenizer, user, target, max_length=cfg.max_length, system_prompt=sys_prompt)

    tokenized = windows.map(_map, remove_columns=windows["train"].column_names)
    status = Counter(s for split in tokenized.values() for s in split["status"])
    print(f"Tokenization: {dict(status)} (max_length={cfg.max_length})")
    tokenized = tokenized.filter(lambda ex: ex["status"] != "dropped").remove_columns(
        ["status", "masked_low_signal_tokens", "weighted_recoverable_tokens", "short_target_downweighted_tokens",
         "token_weights"])

    has_eval = "validation" in tokenized
    t = cfg.training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=t.num_train_epochs,
            max_steps=t.max_steps,
            per_device_train_batch_size=t.per_device_train_batch_size,
            per_device_eval_batch_size=t.per_device_eval_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            warmup_steps=t.warmup_ratio,  # float < 1 == ratio in this transformers version (see train.py)
            lr_scheduler_type=t.lr_scheduler_type,
            logging_steps=t.logging_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=t.eval_steps if has_eval else None,
            save_steps=t.save_steps,
            save_total_limit=t.save_total_limit,
            load_best_model_at_end=has_eval,
            metric_for_best_model="eval_loss" if has_eval else None,
            bf16=True,
            gradient_checkpointing=True,
            train_sampling_strategy="group_by_length",
            remove_unused_columns=False,
            report_to=["tensorboard"],
            seed=t.seed,
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=PadCollator(pad_token_id=tokenizer.pad_token_id),
        callbacks=[SyncToHubCallback(cfg)] if cfg.hub.push_to_hub else [],
    )

    if cfg.eval.baseline:
        print("Baseline evaluation (untrained model)...")
        evaluate_on_test(model, tokenizer, cfg, out_dir, tag="_baseline")

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if cfg.eval.validation_generation and "validation" in windows:
        evaluate_validation_edits(model, tokenizer, cfg, windows["validation"], out_dir)
    if cfg.eval.test_dataset_id:
        final = evaluate_on_test(model, tokenizer, cfg, out_dir)
        print(f"Final (main test set): wer={final['wer']:.4f}, net_fixed_lenient={final['net_fixed_lenient']}, "
              f"broken_per_100_correct={final['broken_per_100_correct']:.2f}")
    if cfg.hub.push_to_hub:
        sync_output_dir(cfg, commit_message="final")


if __name__ == "__main__":
    main()
