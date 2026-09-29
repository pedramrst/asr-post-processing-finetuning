#!/usr/bin/env python3
"""Report on a trained verifier run from its uploaded test predictions --
no GPU, no retraining.

1. Re-filters the candidates with the current normalize.is_style_edit (which
   now also drops clitic respellings like "همینو" -> "همین رو") and recomputes
   the comparison table -- rules only, accept all, oracle, verifier at each
   threshold -- with both the plain lenient metrics and the clitic-normalized
   ones (scoring.py's *_clitic), per slice.
2. Judged precision: the reference is Soniox, so WER can't say whether an
   applied change is a real fix. Candidates are binned by verifier score;
   up to --per-bin per bin are sampled and ruled on by the calibrated judge
   cascade (generate_targets.judge; cached answers are reused). A
   threshold's precision is the bins above it, weighted by their size;
   "real fixes" estimates how many applied changes are genuine corrections.

Writes report.md / report.json / judged.jsonl to --output-dir and uploads
them to a new <run-folder>/report-<time>/ subfolder (unless --no-push).

Example:
  python3 src/edit/verifier_report.py --run-folder verifier-qwen3.5-2b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from audit_verifier import as_item  # noqa: E402
from build_diffs import is_filler_edit  # noqa: E402
from edits import Edit, changes_number  # noqa: E402
from generate_targets import CHEAP_JUDGE, MAIN_JUDGE, Progress, decide, judge  # noqa: E402
from llm_judge import cache_path, load_cache  # noqa: E402
from normalize import is_style_edit  # noqa: E402
from scoring import score  # noqa: E402
from train_edit import load_edit_config, load_test_set  # noqa: E402
from train_verifier import DEFAULTS, apply_accepted  # noqa: E402

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hub-repo", default="PedramR/ASR_Post-processing")
    p.add_argument("--run-folder", default="verifier-qwen3.5-2b")
    p.add_argument("--config", default="configs/edit/qwen3.5-2b-verifier.yaml")
    p.add_argument("--bins", default="0,0.3,0.5,0.7,0.8,0.9",
                   help="Lower edges of the score bins; each is also a threshold in the report.")
    p.add_argument("--per-bin", type=int, default=60)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--judge-dir", default="outputs/edit/judge")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output-dir", default=None, help="Default: outputs/edit/verifier_report/<run-folder>")
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()


def still_candidate(c: dict) -> bool:
    e = Edit(c["start"], c["end"], c["original"].split(), c["proposed"].split())
    return not is_style_edit(e) and not is_filler_edit(e) and not changes_number(e.original, e.replacement)


def main():
    args = parse_args()
    cfg = load_edit_config(args.config, [], defaults=DEFAULTS)
    out = Path(args.output_dir or f"outputs/edit/verifier_report/{args.run_folder}")
    out.mkdir(parents=True, exist_ok=True)
    edges = [float(x) for x in args.bins.split(",")]

    preds = [json.loads(line) for line in open(hf_hub_download(args.hub_repo, f"{args.run_folder}/test_eval/predictions.jsonl"),
                                              encoding="utf-8")]
    ts = load_test_set(cfg)
    by_text = {w: i for i, w in enumerate(ts.whisper)}
    cands = [[] for _ in ts.rows]
    dropped = 0
    for p in preds:
        i = by_text[p["input"]]
        for c in p["candidates"]:
            if still_candidate(c):
                cands[i].append(c)
            else:
                dropped += 1
    flat = [(i, c) for i, cs in enumerate(cands) for c in cs]
    print(f"{len(flat)} candidates after the updated filter ({dropped} clitic/style respellings dropped)")

    # --- judged precision, stratified by score bin
    def bin_of(s):
        return max(k for k, lo in enumerate(edges) if s >= lo)

    rng = random.Random(args.seed)
    by_bin = {k: [] for k in range(len(edges))}
    for i, c in flat:
        by_bin[bin_of(c["p_yes"])].append((i, c))
    sampled = {k: rng.sample(v, min(len(v), args.per_bin)) for k, v in by_bin.items()}
    items = {k: [as_item(ts.rows[i] | {"call_id": ts.rows[i]["call_id"]}, c) for i, c in v] for k, v in sampled.items()}
    all_items = [it for v in items.values() for it in v]
    stats = judge(SimpleNamespace(judge_dir=args.judge_dir, think_max_tokens=8000, workers=args.workers),
                  all_items, Progress(out / "progress.json"))
    cheap = load_cache(cache_path(Path(args.judge_dir), CHEAP_JUDGE))
    main_ = load_cache(cache_path(Path(args.judge_dir), MAIN_JUDGE))
    bin_stats = {}
    with open(out / "judged.jsonl", "w", encoding="utf-8") as f:
        for k, v in items.items():
            c = Counter()
            for it in v:
                status, why = decide(it, cheap, main_, {})
                c[status] += 1
                f.write(json.dumps({"bin": edges[k], **it, "judge_status": status, "judge_reason": why}, ensure_ascii=False) + "\n")
            decided = c["accepted"] + c["rejected"]
            bin_stats[k] = {"candidates": len(by_bin[k]), "sampled": len(v), **c,
                            "precision": c["accepted"] / decided if decided else None}

    def judged_precision(t_index):
        num = den = 0.0
        for k in range(t_index, len(edges)):
            b = bin_stats[k]
            if b["precision"] is not None:
                num += b["candidates"] * b["precision"]
                den += b["candidates"]
        return (num / den if den else None), num

    # --- comparison table
    methods = {"rules only": lambda c: False, "oracle": lambda c: c["effect"] > 0}
    methods.update({("accept all" if lo == 0 else f"verifier@{lo}"): (lambda c, lo=lo: c["p_yes"] >= lo) for lo in edges})
    hyps = {m: [apply_accepted(ts.after_rules[i], cands[i], keep) for i in range(len(ts.rows))] for m, keep in methods.items()}
    report = {"dropped_respellings": dropped, "candidates": len(flat), "bins": {str(edges[k]): v for k, v in bin_stats.items()},
              "judge_cost_usd": round(sum(s["cost_usd"] for s in stats.values()), 4), "slices": {}}
    fmt = lambda x, d=4: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = [f"# Verifier report: {args.run_folder}\n",
             f"{len(flat)} candidates after dropping {dropped} clitic/style respellings. Judge cost ${report['judge_cost_usd']:.2f}. "
             "Judged precision: share of applied changes the calibrated judges call real errors (undecided excluded), "
             "estimated from a score-stratified sample.\n",
             "| slice | method | WER | WER (clitic) | net | net (clitic) | broken/100 | broken/100 (clitic) | entity fix | applied | judged precision | est. real fixes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, idx in ts.slices.items():
        pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
        report["slices"][name] = {}
        for m in ["whisper", "rules only", "accept all"] + [f"verifier@{lo}" for lo in edges if lo > 0] + ["oracle"]:
            h = ts.whisper if m == "whisper" else hyps[m]
            s = score(pick(ts.whisper), pick(h), pick(ts.refs), ts.freq, pick(ts.spans))
            applied = 0 if m in ("whisper", "rules only") else sum(methods[m](c) for i in idx for c in cands[i])
            prec, real = (None, None)
            if name == "main" and (m == "accept all" or m.startswith("verifier@")):
                t_index = 0 if m == "accept all" else edges.index(float(m.split("@")[1]))
                prec, real = judged_precision(t_index)
            s.update(applied=applied, judged_precision=prec, est_real_fixes=real)
            report["slices"][name][m] = s
            lines.append(f"| {name} | {m} | {s['wer']:.4f} | {s['wer_clitic']:.4f} | {s['net_fixed_lenient']} | "
                         f"{s['net_fixed_clitic']} | {s['broken_per_100_correct']:.2f} | {s['broken_per_100_correct_clitic']:.2f} | "
                         f"{(s.get('entity_fix_rate') or 0):.3f} | {applied} | {fmt(prec, 2)} | "
                         f"{'–' if real is None else round(real)} |")
    lines += ["\nScore bins (main set):\n", "| bin | candidates | sampled | real error | not an error | undecided | precision |",
              "|---|---|---|---|---|---|---|"]
    for k, b in bin_stats.items():
        lines.append(f"| >= {edges[k]} | {b['candidates']} | {b['sampled']} | {b.get('accepted', 0)} | {b.get('rejected', 0)} | "
                     f"{b.get('undecided', 0)} | {fmt(b['precision'], 2)} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n".join(lines))
    if not args.no_push:
        from hub_utils import upload_new_folder

        print("Uploaded to", upload_new_folder(args.hub_repo, out, f"{args.run_folder}/report-{time.strftime('%Y%m%d-%H%M')}",
                                               "verifier report"))


if __name__ == "__main__":
    main()
