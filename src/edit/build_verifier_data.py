#!/usr/bin/env python3
"""Step 4a: training data for the edit verifier -- a model that, given a
stretch of Whisper text and ONE proposed change, answers YES (apply it) or
NO (keep Whisper's words).

Why a verifier: the step-3 edit model had to find and fix errors from
Whisper's text alone and couldn't, while the LLM judges did well because they
chose between two given versions. The fine-tuned rewrite model already
proposes candidate changes (most of its edits help, but a large minority
hurt), so the missing piece is deciding which to keep.

Every Whisper-vs-Soniox difference the judges ruled on (generate_targets.py's
two runs) is one example, labelled by the same calibrated rule used for the
training windows: accepted -> YES, confidently rejected (either version is
fine / Whisper right) -> NO, undecided -> left out. Hand labels override.

Splits:
  calibration  the 145 hand-labelled calibration items -- the verifier is
               scored against a person here; their calls are excluded from
               train/validation so nothing from those calls is seen in training
  validation   the calls generate_targets.py put in validation
  train        everything else

Writes outputs/edit/verifier/{train,validation,calibration}.jsonl and
summary.json; --push-folder also uploads them into a new folder of the
shared repo (never overwriting an existing one).

Example:
  python3 src/edit/build_verifier_data.py
  python3 src/edit/build_verifier_data.py --push-folder datasets/verifier-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from calibrate import merge_labels  # noqa: E402
from generate_targets import CHEAP_JUDGE, MAIN_JUDGE, decide  # noqa: E402
from llm_judge import cache_path, load_cache, to_label  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dirs", default="outputs/edit/data,outputs/edit/data_v2")
    p.add_argument("--judge-dir", default="outputs/edit/judge")
    p.add_argument("--calibration-items", default="outputs/edit/targets/calibration_items.jsonl")
    p.add_argument("--labels", default="outputs/edit/targets/calibration_labels.json,"
                   "outputs/edit/targets/calibration_labels_review.json")
    p.add_argument("--output-dir", default="outputs/edit/verifier")
    p.add_argument("--push-folder", default=None, metavar="FOLDER",
                   help="Also upload the split files into this NEW folder of --push-repo (e.g. datasets/verifier-v1); "
                        "refuses if the folder already exists. The repo is public and the files contain call text.")
    p.add_argument("--push-repo", default="PedramR/ASR_Post-processing")
    p.add_argument("--validation-frac", type=float, default=0.05,
                   help="For calls not in any train.jsonl split (their windows were excluded as undecided).")
    return p.parse_args()


def example(item: dict, label: bool, source: str) -> dict:
    return {"id": item["id"], "call_id": item["call_id"], "stratum": item["stratum"], "crm_names": item.get("crm_names") or [],
            "left": item["left"], "original": item["whisper_span"], "proposed": item["soniox_span"], "right": item["right"],
            "label": "YES" if label else "NO", "label_source": source}


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    judge_dir = Path(args.judge_dir)
    cheap = load_cache(cache_path(judge_dir, CHEAP_JUDGE))
    main_ = load_cache(cache_path(judge_dir, MAIN_JUDGE))
    human = merge_labels(args.labels.split(","))

    # calibration split: hand-labelled items
    calibration, calib_calls = [], set()
    for line in open(args.calibration_items, encoding="utf-8"):
        it = json.loads(line)
        if it["id"] not in human or it["kind"] != "substitute":
            continue
        label = to_label(it, human[it["id"]]["verdict"])
        calib_calls.add(it["call_id"])
        if label == "real_error":
            calibration.append(example(it, True, "hand label"))
        elif label in ("either_fine", "soniox_wrong"):
            calibration.append(example(it, False, "hand label"))

    # call -> split from the training windows
    call_split = {}
    items = {}
    for d in args.data_dirs.split(","):
        for line in open(Path(d) / "train.jsonl", encoding="utf-8"):
            w = json.loads(line)
            call_split[w["call_id"]] = w["split"]
        for line in open(Path(d) / "judge_items.jsonl", encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it

    def split_of(call_id: str) -> str:
        if call_id in call_split:
            return call_split[call_id]
        h = int(hashlib.sha1(call_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "validation" if h < args.validation_frac else "train"

    splits = {"train": [], "validation": []}
    counts = Counter()
    for it in items.values():
        if it["kind"] != "substitute":
            counts["skipped: not a substitution"] += 1
            continue
        if it["call_id"] in calib_calls:
            counts["skipped: call is in the calibration split"] += 1
            continue
        status, why = decide(it, cheap, main_, {})  # hand-labelled calls are all in calibration
        if status == "undecided":
            counts["skipped: undecided"] += 1
            continue
        splits[split_of(it["call_id"])].append(example(it, status == "accepted", "judges"))

    splits["calibration"] = calibration
    summary = {"skipped": dict(counts)}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = {"examples": len(rows), "YES": sum(r["label"] == "YES" for r in rows),
                         "NO": sum(r["label"] == "NO" for r in rows), "calls": len({r["call_id"] for r in rows}),
                         "by_stratum": dict(Counter(r["stratum"] for r in rows))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.push_folder:
        from hub_utils import upload_new_folder

        print("Uploaded to", upload_new_folder(args.push_repo, out, args.push_folder, "verifier training data"))


if __name__ == "__main__":
    main()
