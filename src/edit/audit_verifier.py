#!/usr/bin/env python3
"""Audit a verifier run's disagreements with the reference, using the same
calibrated LLM-judge rule that labelled its training data.

The test metric scores candidates against Soniox's reference, which rewards
matching Soniox's wording even where Whisper's was fine. So when the
verifier rejects a candidate the reference calls helpful, it's either a real
miss or a correct rejection of a style-only change. This samples:

  rejected_helpful  reference says it helps, verifier score < --reject-below
  accepted_harmful  reference says it hurts, verifier score >= --accept-above

and has the judges (MiMo, then DeepSeek where MiMo doesn't rule it out) rule
on each, blind, exactly like generate_targets.py. Writes audit.md and
audit.jsonl to --output-dir.

Example:
  python3 src/edit/audit_verifier.py --run-folder verifier-qwen3.5-2b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from generate_targets import CHEAP_JUDGE, MAIN_JUDGE, Progress, decide, judge  # noqa: E402
from llm_judge import cache_path, load_cache  # noqa: E402

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hub-repo", default="PedramR/ASR_Post-processing")
    p.add_argument("--run-folder", default="verifier-qwen3.5-2b")
    p.add_argument("--reject-below", type=float, default=0.3)
    p.add_argument("--accept-above", type=float, default=0.5)
    p.add_argument("--n-rejected-helpful", type=int, default=200)
    p.add_argument("--n-accepted-harmful", type=int, default=100)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--judge-dir", default="outputs/edit/judge")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output-dir", default="outputs/edit/verifier_audit")
    return p.parse_args()


def as_item(row: dict, c: dict) -> dict:
    """A candidate in llm_judge.py's difference-item format (Whisper side =
    Whisper after rules, the "Soniox" side = the rewrite model's proposal)."""
    iid = hashlib.sha1(f"audit\x1f{row['call_id']}\x1f{c['start']}\x1f{c['end']}\x1f{c['proposed']}".encode()).hexdigest()[:16]
    return {"id": iid, "call_id": row["call_id"], "whisper_span": c["original"], "soniox_span": c["proposed"],
            "left": c["left"], "right": c["right"], "crm_names": c.get("crm_names") or [], "kind": "substitute",
            "stratum": "audit", "p_yes": c["p_yes"], "effect": c["effect"]}


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(hf_hub_download(args.hub_repo, f"{args.run_folder}/test_eval/predictions.jsonl"),
                                             encoding="utf-8")]
    groups = {"rejected_helpful": [], "accepted_harmful": []}
    for r in rows:
        for c in r["candidates"]:
            if c["effect"] > 0 and c["p_yes"] < args.reject_below:
                groups["rejected_helpful"].append(as_item(r, c))
            elif c["effect"] < 0 and c["p_yes"] >= args.accept_above:
                groups["accepted_harmful"].append(as_item(r, c))
    rng = random.Random(args.seed)
    sample = {g: rng.sample(v, min(len(v), n)) for (g, v), n in
              zip(groups.items(), (args.n_rejected_helpful, args.n_accepted_harmful))}
    print({g: f"{len(sample[g])} sampled of {len(v)}" for g, v in groups.items()})
    items = [it for v in sample.values() for it in v]

    jargs = SimpleNamespace(judge_dir=args.judge_dir, think_max_tokens=8000, workers=args.workers)
    stats = judge(jargs, items, Progress(out / "progress.json"))
    cheap = load_cache(cache_path(Path(args.judge_dir), CHEAP_JUDGE))
    main_ = load_cache(cache_path(Path(args.judge_dir), MAIN_JUDGE))

    lines = [f"# Verifier audit: {args.run_folder}\n",
             f"Judge cost: ${sum(s['cost_usd'] for s in stats.values()):.2f}\n",
             "| group | n | judges: real error (accept) | judges: no error (reject) | undecided |", "|---|---|---|---|---|"]
    with open(out / "audit.jsonl", "w", encoding="utf-8") as f:
        for g, v in sample.items():
            c = Counter()
            for it in v:
                status, why = decide(it, cheap, main_, {})
                c[status] += 1
                f.write(json.dumps({"group": g, **it, "judge_status": status, "judge_reason": why}, ensure_ascii=False) + "\n")
            n = len(v)
            lines.append(f"| {g} | {n} | {c['accepted']} ({c['accepted'] / max(n, 1):.0%}) | "
                         f"{c['rejected']} ({c['rejected'] / max(n, 1):.0%}) | {c['undecided']} ({c['undecided'] / max(n, 1):.0%}) |")
    (out / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
