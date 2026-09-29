#!/usr/bin/env python3
"""Score correct.py's output against the reference, at any threshold.

correct.py writes every candidate with its verifier score (with
--keep-intermediates), so this re-derives the corrected text for as many
thresholds as asked for without re-running either model -- the generation
pass is the expensive part and it's already done.

Reported per slice (main / entity / typo), next to the same baselines the
rest of the project uses: untouched Whisper, rules only, and accepting
every candidate. Scored with scoring.py, so the numbers line up with
train_verifier.py's and verifier_report.py's tables.

  python3 src/edit/score_corrected.py --corrected outputs/corrected-full.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from dotenv import load_dotenv  # noqa: E402

from edits import Edit, apply_edits, validate_edits  # noqa: E402
from scoring import score  # noqa: E402
from train_edit import load_edit_config, load_test_set  # noqa: E402
from train_verifier import DEFAULTS  # noqa: E402

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrected", required=True, help="correct.py output, written with --keep-intermediates.")
    p.add_argument("--config", default="configs/edit/qwen3.5-2b-verifier.yaml",
                   help="Only for the eval settings (test set, slices, entity labels).")
    p.add_argument("--thresholds", default="0.3,0.5,0.7,0.8,0.9")
    p.add_argument("--output", default=None, help="Default: <corrected>.scores.md next to the input.")
    return p.parse_args()


def apply_at(after_rules: str, candidates: list[dict], threshold: float) -> tuple[str, int]:
    """(corrected text, number of edits actually applied) -- the count is
    post-validation, so overlapping candidates that get dropped don't count."""
    edits = [Edit(c["start"], c["end"], c["original"].split(), c["proposed"].split())
             for c in candidates if c["score"] >= threshold]
    valid, _ = validate_edits(after_rules.split(), sorted(edits, key=lambda e: e.start))
    return apply_edits(after_rules, valid), len(valid)


def main():
    args = parse_args()
    rows = [json.loads(line) for line in open(args.corrected, encoding="utf-8")]
    if "candidates" not in rows[0]:
        raise SystemExit("this file has no candidates -- rerun correct.py with --keep-intermediates")
    thresholds = [float(t) for t in args.thresholds.split(",")]

    cfg = load_edit_config(args.config, [], defaults=DEFAULTS)
    ts = load_test_set(cfg)
    by_text = {w: i for i, w in enumerate(ts.whisper)}
    # correct.py's rows, lined up with the test set's order
    idx = [by_text[r["input"]] for r in rows if r["input"] in by_text]
    missing = len(rows) - len(idx)
    if missing:
        print(f"{missing} corrected rows aren't in the test set -- skipped")
    row_of = {by_text[r["input"]]: r for r in rows if r["input"] in by_text}

    # each method: row -> (corrected text, edits applied)
    methods = {"whisper": lambda r: (r["input"], 0), "rules only": lambda r: (r["after_rules"], 0),
               "accept all": lambda r: apply_at(r["after_rules"], r["candidates"], -1.0)}
    for t in thresholds:
        methods[f"verifier@{t}"] = (lambda r, t=t: apply_at(r["after_rules"], r["candidates"], t))

    stopped = sum(r.get("premature_stop", False) for r in rows)
    n_cand = sum(len(r["candidates"]) for r in rows)
    lines = [f"# End-to-end corrector: {args.corrected}\n",
             f"{len(idx)} transcripts scored; {n_cand} candidates; "
             f"{stopped} rewrites skipped for stopping early.\n",
             "| slice | method | WER | WER (clitic) | fixed | broken | net | broken/100 | entity fix | applied |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, sl in ts.slices.items():
        sl = [i for i in sl if i in row_of]
        for m, build in methods.items():
            built = [build(row_of[i]) for i in sl]
            hyps = [text for text, _ in built]
            applied = sum(n for _, n in built)
            s = score([ts.whisper[i] for i in sl], hyps, [ts.refs[i] for i in sl], ts.freq,
                      [ts.spans[i] for i in sl])
            lines.append(f"| {name} | {m} | {s['wer']:.4f} | {s['wer_clitic']:.4f} | {s['fixed_lenient']} | "
                         f"{s['broken_lenient']} | {s['net_fixed_lenient']} | {s['broken_per_100_correct']:.2f} | "
                         f"{(s.get('entity_fix_rate') or 0):.3f} | {applied} |")
    out = Path(args.output or (args.corrected + ".scores.md"))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
