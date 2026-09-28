#!/usr/bin/env python3
"""Publish generate_targets.py's training windows as one Hub dataset, so a
GPU instance can train on them (configs/edit/*.yaml point data.sources at
it). Combines any number of train.jsonl files into train/validation splits
(each window's own by-call `split`), and pushes it as a PRIVATE dataset --
it contains real call transcripts.

Example:
  python3 src/edit/publish_edit_dataset.py
  python3 src/edit/publish_edit_dataset.py --files outputs/edit/data/train.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", default="outputs/edit/data/train.jsonl,outputs/edit/data_v2/train.jsonl")
    p.add_argument("--repo-id", default="PedramR/ASR_Post-processing-edit-dataset")
    p.add_argument("--dry-run", action="store_true", help="Build and summarize, don't push.")
    args = p.parse_args()

    rows, seen = [], set()
    for path in args.files.split(","):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["window_id"] in seen:
                raise SystemExit(f"window {r['window_id']} appears twice -- are two files from the same run?")
            seen.add(r["window_id"])
            r["source_file"] = path
            rows.append(r)
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation")}
    calls = {s: {r["call_id"] for r in v} for s, v in splits.items()}
    if calls["train"] & calls["validation"]:
        raise SystemExit("a call is in both splits")
    print({s: {"windows": len(v), "with edits": sum(bool(r["edits"]) for r in v),
               "edits": sum(len(r["edits"]) for r in v)} for s, v in splits.items()},
          dict(Counter(r["source_file"] for r in rows)))
    if args.dry_run:
        return
    DatasetDict({s: Dataset.from_list(v) for s, v in splits.items()}).push_to_hub(args.repo_id, private=True)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id} (private)")


if __name__ == "__main__":
    main()
