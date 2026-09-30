#!/usr/bin/env python3
"""What does the verifier think of the edits the candidate filter drops?

extract_candidates() (train_verifier.py) drops any edit that only changes
spacing (is_style_edit: squash(a) == squash(b)) on the assumption that
rules.py already splits glued words. It doesn't catch them all: the rewrite
model splits "واسهیکی‌شون" and the filter discards it, so the verifier never
sees it. This measures what those edits are worth before deciding whether
to let them through.

  extract   (local, no GPU) the rewrite model's spacing-only edits (words
            joined or split, same characters) from correct.py's
            --keep-intermediates output -> one jsonl of verifier items
  score     (GPU) P(YES) for each with the current verifier adapter
  report    (local) per verifier-score bin: how many, and what applying
            them alone does to the words scored against the reference
            (fixed - broken, lenient), next to the same numbers for the
            candidates the pipeline does send to the verifier

  python3 src/edit/score_dropped.py extract --corrected outputs/edit/e2e/corrected-full.jsonl
  python3 src/edit/score_dropped.py score   --input dropped.jsonl --output dropped.scored.jsonl
  python3 src/edit/score_dropped.py report  --scored dropped.scored.jsonl --corrected ...
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

BINS = [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["extract", "score", "report", "audit"])
    p.add_argument("--corrected", default="outputs/edit/e2e/corrected-full.jsonl")
    p.add_argument("--input", default="outputs/edit/e2e/dropped-candidates.jsonl")
    p.add_argument("--output", default="outputs/edit/e2e/dropped-candidates.scored.jsonl")
    p.add_argument("--scored", default=None)
    p.add_argument("--hub-repo", default="PedramR/ASR_Post-processing")
    p.add_argument("--verifier-run", default="verifier-qwen3.5-2b-v2")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--reference-file", default="qwen3.5-2b-100pct-masked-weighted-v2/test_eval_best/predictions.jsonl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-words", type=int, default=12)
    p.add_argument("--sample", type=int, default=150, help="audit: how many dropped edits to have judged.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--judge-dir", default="outputs/edit/judge")
    p.add_argument("--workers", type=int, default=32)
    return p.parse_args()


def dropped_edits(row: dict, context_words: int) -> list[dict]:
    """Spacing-only substitutions in the rewrite that the candidate filter drops."""
    from edits import extract_edits
    from normalize import squash
    from train_verifier import strip_punctuation

    words = row["after_rules"].split()
    out = []
    for e in extract_edits(row["after_rules"], strip_punctuation(row["rewrite"])):
        a, b = " ".join(e.original), " ".join(e.replacement)
        if (e.kind == "substitute" and max(len(e.original), len(e.replacement)) <= 4
                and a != b and squash(a) == squash(b) and len(e.original) != len(e.replacement)):
            out.append({"start": e.start, "end": e.end, "original": a, "proposed": b,
                        "left": " ".join(words[max(0, e.start - context_words):e.start]),
                        "right": " ".join(words[e.end:e.end + context_words])})
    return out


def extract(args):
    n = 0
    with open(args.input, "w", encoding="utf-8") as f:
        for i, line in enumerate(open(args.corrected, encoding="utf-8")):
            row = json.loads(line)
            if row.get("premature_stop"):
                continue
            # correct.py doesn't write a row's CRM names; its candidates carry them
            names = row["candidates"][0]["crm_names"] if row.get("candidates") else []
            for c in dropped_edits(row, args.context_words):
                c.update(row=i, crm_names=names)
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} spacing-only edits -> {args.input}")


def score(args):
    import torch
    from huggingface_hub import hf_hub_download  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from correct import score_candidates

    items = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, args.hub_repo, subfolder=args.verifier_run)
    model.eval()
    for it, s in zip(items, score_candidates(model, tokenizer, items, args.batch_size)):
        it["score"] = s
    with open(args.output, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} scored -> {args.output}")


def report(args):
    from huggingface_hub import hf_hub_download

    from train_verifier import candidate_effect

    rows = [json.loads(line) for line in open(args.corrected, encoding="utf-8")]
    ref_of = {}
    for line in open(hf_hub_download(args.hub_repo, args.reference_file), encoding="utf-8"):
        r = json.loads(line)
        ref_of[r["input"]] = r["reference"]

    def table(cands, title):
        lines = [f"\n### {title}\n", "| verifier score | edits | fix words | break words | neutral | net words |",
                 "|---|---|---|---|---|---|"]
        for lo, hi in BINS:
            grp = [c for c in cands if lo <= c["score"] < hi]
            eff = [candidate_effect(rows[c["row"]]["after_rules"], ref_of[rows[c["row"]]["input"]], c) for c in grp]
            lines.append(f"| {lo:.1f}-{min(hi, 1.0):.1f} | {len(grp)} | {sum(e > 0 for e in eff)} | "
                         f"{sum(e < 0 for e in eff)} | {sum(e == 0 for e in eff)} | {sum(eff):+d} |")
        return lines

    dropped = [json.loads(line) for line in open(args.scored or args.output, encoding="utf-8")]
    kept = [dict(c, row=i) for i, r in enumerate(rows) for c in r["candidates"]]
    out = ["# Verifier scores on spacing-only edits the filter drops\n",
           "Net words = words fixed minus words broken by applying the edit alone, lenient, against the reference."]
    out += table(dropped, "Dropped by the filter (never reach the verifier)")
    out += table(kept, "Sent to the verifier today (for comparison)")
    text = "\n".join(out) + "\n"
    Path(args.output).with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)


def audit(args):
    """Have the calibrated LLM judges (MiMo cheap-first, DeepSeek main) rule on
    a random sample of the dropped edits: is the glued/split form a real
    error, or are both fine? Crossed with the edit's effect on the reference,
    so 'fixes words' can be told apart from 'matches Soniox's spacing'."""
    import hashlib
    import random
    from collections import Counter
    from types import SimpleNamespace

    from huggingface_hub import hf_hub_download

    from generate_targets import CHEAP_JUDGE, MAIN_JUDGE, Progress, decide, judge
    from llm_judge import cache_path, load_cache
    from train_verifier import candidate_effect

    rows = [json.loads(line) for line in open(args.corrected, encoding="utf-8")]
    ref_of = {}
    for line in open(hf_hub_download(args.hub_repo, args.reference_file), encoding="utf-8"):
        r = json.loads(line)
        ref_of[r["input"]] = r["reference"]
    cands = [json.loads(line) for line in open(args.scored or args.output, encoding="utf-8")]
    picked = random.Random(args.seed).sample(cands, min(args.sample, len(cands)))
    items = []
    for c in picked:
        row = rows[c["row"]]
        iid = hashlib.sha1(f"spacing\x1f{c['row']}\x1f{c['start']}\x1f{c['end']}\x1f{c['proposed']}".encode()).hexdigest()[:16]
        items.append({"id": iid, "call_id": str(c["row"]), "whisper_span": c["original"], "soniox_span": c["proposed"],
                      "left": c["left"], "right": c["right"], "crm_names": c.get("crm_names") or [], "kind": "substitute",
                      "stratum": "spacing", "effect": candidate_effect(row["after_rules"], ref_of[row["input"]], c),
                      "score": c["score"]})
    out = Path(args.output).with_suffix(".audit")
    out.mkdir(parents=True, exist_ok=True)
    judge(SimpleNamespace(judge_dir=args.judge_dir, think_max_tokens=8000, workers=args.workers),
          items, Progress(out / "progress.json"))
    cheap = load_cache(cache_path(Path(args.judge_dir), CHEAP_JUDGE))
    main_ = load_cache(cache_path(Path(args.judge_dir), MAIN_JUDGE))
    table, lines = {}, []
    with open(out / "judged.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            status, why = decide(it, cheap, main_, {})
            kind = "fixes words" if it["effect"] > 0 else "breaks words" if it["effect"] < 0 else "neutral"
            table.setdefault(kind, Counter())[status] += 1
            f.write(json.dumps({**it, "judge_status": status, "judge_reason": why}, ensure_ascii=False) + "\n")
    tot = Counter(s for c in table.values() for s in c.elements())
    decided = tot["accepted"] + tot["rejected"]
    lines = [f"# Judge audit of {len(items)} sampled spacing-only edits\n",
             "accepted = judges call the original a real error; rejected = both forms fine / original right.\n",
             "| effect vs reference | edits | real error | not an error | undecided |", "|---|---|---|---|---|"]
    for kind in ("fixes words", "breaks words", "neutral"):
        c = table.get(kind, Counter())
        lines.append(f"| {kind} | {sum(c.values())} | {c['accepted']} | {c['rejected']} | {c['undecided']} |")
    lines.append(f"| **all** | {len(items)} | {tot['accepted']} | {tot['rejected']} | {tot['undecided']} |")
    lines.append(f"\nJudged precision (undecided excluded): {tot['accepted'] / decided:.2f}" if decided else "")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    a = parse_args()
    {"extract": extract, "score": score, "report": report, "audit": audit}[a.mode](a)
