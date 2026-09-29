#!/usr/bin/env python3
"""Threshold decoding for a trained edit model -- no retraining.

Greedy decoding is biased towards NONE: its whole "nothing to fix"
probability sits on one first token, while an edit's probability is split
across many different word numbers, so NONE usually wins even when the model
thinks an edit is fairly likely. This measures, per window, p_none = the
model's probability of NONE as the first output token, and generates each
window twice: greedy, and with NONE blocked at the first step (the model's
best edit list). At threshold t a window uses the blocked output when
p_none < t, else the greedy one; t = 0 is plain greedy decoding.

Reported for a sweep of thresholds:
  * validation windows: exact-match edit precision/recall against the
    verified targets, plus how well p_none separates windows that need an
    edit from ones that don't (AUC);
  * the first --test-transcripts test transcripts through the full pipeline
    (rules, overlapping windows, merge), scored like train_edit.py's
    evaluation: WER, words fixed/broken, entity fix rate.

Writes sweep.md / sweep.json and per-window records (p_none, greedy and
blocked outputs) to --output-dir, and uploads them into the run's Hub
folder under decode_sweep/ (unless --no-push).

Example (on the GPU instance):
  python3 src/edit/decode_sweep.py --run-folder edit-qwen3.5-2b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from peft import PeftModel  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList  # noqa: E402

from edit_format import NONE, render_input, system_prompt  # noqa: E402
from evaluate import generate_batch  # noqa: E402
from scoring import score  # noqa: E402
from train_edit import (  # noqa: E402
    chat_prompts, load_edit_config, load_test_set, load_windows, validation_scores, window_records,
)
from windowing import build_window_prompts, merge_outputs  # noqa: E402

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-folder", default="edit-qwen3.5-2b", help="The run's folder in --hub-repo (its adapter is the best checkpoint).")
    p.add_argument("--hub-repo", default="PedramR/ASR_Post-processing")
    p.add_argument("--adapter-dir", default=None, help="Local adapter dir instead of the Hub folder.")
    p.add_argument("--config", default="configs/edit/qwen3.5-2b-edit.yaml", help="For model_id, data sources, eval settings.")
    p.add_argument("--test-transcripts", type=int, default=300)
    p.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default=None, help="Default: outputs/edit-decode-sweep/<run-folder>")
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()


class BlockFirstToken(LogitsProcessor):
    """Forbids `token_id` as the first generated token only."""

    def __init__(self, prompt_len: int, token_id: int):
        self.prompt_len, self.token_id = prompt_len, token_id

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            scores[:, self.token_id] = float("-inf")
        return scores


def _batches(prompts, batch_size):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    for i in range(0, len(order), batch_size):
        yield order[i:i + batch_size]


@torch.no_grad()
def first_token_prob(model, tokenizer, prompts: list[str], token_id: int, batch_size: int) -> list[float]:
    """P(first generated token == token_id) per prompt, from one forward pass
    that keeps only the last position's logits."""
    probs = [0.0] * len(prompts)
    tokenizer.padding_side = "left"
    for idx in tqdm(list(_batches(prompts, batch_size)), desc="p_none"):
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        try:
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        except TypeError:  # older model classes without logits_to_keep
            logits = model(**enc).logits[:, -1, :]
        p = torch.softmax(logits.float(), dim=-1)[:, token_id].tolist()
        for i, v in zip(idx, p):
            probs[i] = v
    return probs


@torch.no_grad()
def generate_blocked(model, tokenizer, prompts: list[str], token_id: int, max_new_tokens: int, batch_size: int) -> list[str]:
    """Greedy decoding with token_id forbidden at the first step."""
    outs = [""] * len(prompts)
    tokenizer.padding_side = "left"
    for idx in tqdm(list(_batches(prompts, batch_size)), desc="blocked NONE"):
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        n = enc["input_ids"].shape[1]
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                             logits_processor=LogitsProcessorList([BlockFirstToken(n, token_id)]))
        for i, row in zip(idx, gen[:, n:].tolist()):
            while row and row[-1] == tokenizer.pad_token_id:
                row.pop()
            outs[i] = tokenizer.decode(row, skip_special_tokens=True)
    return outs


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a random `pos` score < a random `neg` score): how often a window that
    needs an edit gets a lower p_none than one that doesn't."""
    if not pos or not neg:
        return None
    wins = sum((p < n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def choose(p_none, greedy, blocked, t):
    return [b if p < t else g for p, g, b in zip(p_none, greedy, blocked)]


def main():
    args = parse_args()
    cfg = load_edit_config(args.config, [f"eval.max_examples={args.test_transcripts}"])
    out = Path(args.output_dir or f"outputs/edit-decode-sweep/{args.run_folder}")
    out.mkdir(parents=True, exist_ok=True)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16, device_map="auto")
    model = (PeftModel.from_pretrained(base, args.adapter_dir) if args.adapter_dir
             else PeftModel.from_pretrained(base, args.hub_repo, subfolder=args.run_folder))
    model.eval()
    none_id = tokenizer(NONE, add_special_tokens=False)["input_ids"][0]
    sys_prompt = system_prompt(cfg.data.target_includes_original)
    e = cfg.eval

    def run(users):
        prompts = chat_prompts(tokenizer, users, sys_prompt)
        p_none = first_token_prob(model, tokenizer, prompts, none_id, args.batch_size)
        greedy = generate_batch(model, tokenizer, prompts, e.max_new_tokens, args.batch_size)
        need = [i for i, p in enumerate(p_none) if p < max(thresholds)]
        blocked = list(greedy)
        for i, o in zip(need, generate_blocked(model, tokenizer, [prompts[i] for i in need], none_id,
                                               e.max_new_tokens, args.batch_size)):
            blocked[i] = o
        return p_none, greedy, blocked

    # --- validation windows
    val = load_windows(cfg)["validation"]
    v_p, v_g, v_b = run([render_input(r["source"].split(), r.get("crm_names") or []) for r in val])
    has_edit = [bool(r["edits"]) for r in val]
    report = {"validation": {"p_none_auc": auc([p for p, h in zip(v_p, has_edit) if h], [p for p, h in zip(v_p, has_edit) if not h]),
                             "mean_p_none_with_edit": sum(p for p, h in zip(v_p, has_edit) if h) / max(sum(has_edit), 1),
                             "mean_p_none_without_edit": sum(p for p, h in zip(v_p, has_edit) if not h) / max(len(has_edit) - sum(has_edit), 1),
                             "by_threshold": {}}}
    for t in [0.0] + thresholds:
        m, _ = validation_scores(val, choose(v_p, v_g, v_b, t))
        report["validation"]["by_threshold"][str(t)] = m
    with open(out / "validation_windows.jsonl", "w", encoding="utf-8") as f:
        for r, p, g, b in zip(val, v_p, v_g, v_b):
            f.write(json.dumps({"window_id": r["window_id"], "source": r["source"], "gold_edits": r["edits"],
                                "p_none": p, "greedy": g, "blocked": b}, ensure_ascii=False) + "\n")

    # --- test transcripts, full pipeline
    ts = load_test_set(cfg)
    index, users = build_window_prompts(ts.after_rules, ts.names, e.window_words, e.stride)
    print(f"Test: {len(ts.rows)} transcripts, {len(users)} windows")
    t_p, t_g, t_b = run(users)
    report["test"] = {"transcripts": len(ts.rows), "windows": len(users), "by_threshold": {}}
    rules_scores = {name: score([ts.whisper[i] for i in idx], [ts.after_rules[i] for i in idx], [ts.refs[i] for i in idx],
                                ts.freq, [ts.spans[i] for i in idx]) for name, idx in ts.slices.items()}
    report["test"]["rules"] = rules_scores
    for t in [0.0] + thresholds:
        results = merge_outputs(ts.after_rules, index, choose(t_p, t_g, t_b, t))
        per_slice = {}
        for name, idx in ts.slices.items():
            m = score([ts.whisper[i] for i in idx], [results[i].corrected for i in idx], [ts.refs[i] for i in idx],
                      ts.freq, [ts.spans[i] for i in idx])
            m["model_edits"] = sum(len(results[i].edits) for i in idx)
            per_slice[name] = m
        report["test"]["by_threshold"][str(t)] = per_slice
    with open(out / "test_windows.jsonl", "w", encoding="utf-8") as f:
        for t_idx, recs in enumerate(window_records(index, t_g, len(ts.rows), p_none=t_p, blocked=t_b)):
            f.write(json.dumps({"call_id": ts.rows[t_idx]["call_id"], "after_rules": ts.after_rules[t_idx],
                                "reference": ts.refs[t_idx], "windows": recs}, ensure_ascii=False) + "\n")

    # --- tables
    v = report["validation"]
    lines = [f"# Threshold decoding: {args.run_folder}\n",
             f"Validation ({len(val)} windows): p_none AUC {v['p_none_auc']:.3f}; mean p_none "
             f"{v['mean_p_none_with_edit']:.3f} on windows needing an edit vs {v['mean_p_none_without_edit']:.3f} on others.\n",
             "| threshold | edit precision | edit recall | predicted | gold | rejected lines |", "|---|---|---|---|---|---|"]
    for t, m in v["by_threshold"].items():
        fmt = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(f"| {t} | {fmt(m['edit_precision'])} | {fmt(m['edit_recall'])} | {m['predicted_edits']} | "
                     f"{m['gold_edits']} | {m['rejected_lines']} |")
    lines += [f"\nTest, first {len(ts.rows)} transcripts (threshold 0.0 = greedy):\n",
              "| slice | threshold | WER | fixed | broken | net | broken/100 | entity fix | model edits |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in ts.slices:
        r = rules_scores[name]
        lines.append(f"| {name} | rules only | {r['wer']:.4f} | {r['fixed_lenient']} | {r['broken_lenient']} | "
                     f"{r['net_fixed_lenient']} | {r['broken_per_100_correct']:.2f} | {(r.get('entity_fix_rate') or 0):.3f} | 0 |")
        for t, per_slice in report["test"]["by_threshold"].items():
            m = per_slice[name]
            lines.append(f"| {name} | {t} | {m['wer']:.4f} | {m['fixed_lenient']} | {m['broken_lenient']} | "
                         f"{m['net_fixed_lenient']} | {m['broken_per_100_correct']:.2f} | "
                         f"{(m.get('entity_fix_rate') or 0):.3f} | {m['model_edits']} |")
    (out / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "sweep.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n".join(lines))

    if not args.no_push and not args.adapter_dir:
        HfApi().upload_folder(repo_id=args.hub_repo, folder_path=str(out),
                              path_in_repo=f"{args.run_folder}/decode_sweep", commit_message="decode sweep")
        print(f"Uploaded to https://huggingface.co/{args.hub_repo}/tree/main/{args.run_folder}/decode_sweep")


if __name__ == "__main__":
    main()
