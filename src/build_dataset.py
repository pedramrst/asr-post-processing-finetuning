#!/usr/bin/env python3
"""Build ASR-correction training data from callcc-2k.

Streams each parquet shard directly off the Hub -- a call's segments always
live entirely within one shard, so no cross-shard joins are needed. Each call
is deterministically assigned to either "assembled" (one row per channel,
holding that channel's full turn sequence) or "chunked" (one row per
segment) based on --seed/--assembled-ratio.

Assembled rows are per-channel, not per-call with both channels merged: the
callcc-test-1k benchmark is structured the same way (one row per call
channel), and a merged-channel row doesn't correspond to anything actually
corrected in one pass in production -- each speaker's channel is handled on
its own.

Production runs correction over one full call channel at a time (confirmed,
not just inferred from callcc-test-1k's row shape) -- so assembled rows are
the primary training signal, not a minority, and --assembled-ratio should
normally be high (see the flag description below). Chunked rows are kept as
a smaller supplementary set: they're what carries prepare_split.py's
low-WER "already correct, leave it alone" signal, since only chunked rows
get a `bucket` label.

Shards are read column-pruned over HTTP via HfFileSystem instead of being
downloaded whole: `audio` is embedded in the same parquet files as the text
columns we need, and a plain snapshot_download would pull every shard's
embedded audio bytes to disk (100s of GB) before any column filtering could
happen. Reading with `columns=SEG_COLS` through a random-access remote file
object only fetches the byte ranges for those columns' data.

Requires being logged in already (`huggingface-cli login`) or an HF_TOKEN set
in the environment or a `.env` file (see .env.example) -- this script does
not take or store a token itself.

Flags:
  --assembled-ratio FLOAT   Required. Fraction (0-1) of calls emitted as a
                            single assembled conversation; the rest are
                            emitted as one row per segment ("chunked"). Since
                            production corrects one full channel at a time,
                            this should normally be high (e.g. 0.9) so most
                            of the raw data is even eligible to become
                            assembled rows -- prepare_split.py's
                            --assembled-target-frac can only ever downsample
                            assembled rows relative to chunked ones, never
                            manufacture more, so it can't fix an insufficient
                            supply of them created here.
  --seed INT                Optional, default 42. Seeds the deterministic
                            per-call assembled/chunked assignment; same seed
                            + ratio always yields the same split.
  --output-dir PATH          Optional. Full output path, including filename.
                            Defaults to ./asr_dataset_<timestamp>.jsonl.
  --max-calls INT            Optional. Stop after this many calls (testing).
  --max-shards INT           Optional. Only scan this many shard files
                            (testing).

Example:
  python3 src/build_dataset.py --assembled-ratio 0.9
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from tqdm import tqdm

load_dotenv()

HF_REPO_ID = "ErfanRou/callcc-2k"

SEG_COLS = [
    "id", "call_id", "channel", "start", "end", "duration",
    "text", "text_raw", "text_whisper",
    "conf_mean", "conf_min", "wer_whisper", "bucket",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--assembled-ratio", type=float, required=True,
                    help="Fraction of calls (0-1) emitted as fully assembled conversations; rest are chunked per-segment.")
    p.add_argument("--output-dir", default=None,
                    help="Full path (including filename) where the output .jsonl is written. "
                         "If omitted, defaults to ./asr_dataset_<timestamp>.jsonl")
    p.add_argument("--max-calls", type=int, default=None, help="Cap total calls processed (testing).")
    p.add_argument("--max-shards", type=int, default=None, help="Cap number of shard files scanned (testing).")
    return p.parse_args()


def list_shard_paths(fs):
    return sorted(fs.glob(f"datasets/{HF_REPO_ID}/data/*.parquet"))


def read_remote_table(fs, path, columns):
    with fs.open(path, "rb") as f:
        return pq.ParquetFile(f).read(columns=columns, use_threads=True)


def load_crm_and_expected(fs):
    """Returns (crm_map, expected) where expected[call_id] is {channel: expected_segment_count}."""
    meta_path = f"datasets/{HF_REPO_ID}/meta/calls_meta.parquet"
    tbl = read_remote_table(fs, meta_path, ["id", "client_metadata_json", "n_seg_c0", "n_seg_c1"])
    ids = tbl.column("id").to_pylist()
    cmjs = tbl.column("client_metadata_json").to_pylist()
    nc0 = tbl.column("n_seg_c0").to_pylist()
    nc1 = tbl.column("n_seg_c1").to_pylist()
    crm_map, expected = {}, {}
    for i, cid in enumerate(ids):
        expected[cid] = {0: nc0[i] or 0, 1: nc1[i] or 0}
        j = cmjs[i]
        if not j:
            crm_map[cid] = None
            continue
        try:
            d = json.loads(j)
        except Exception:
            crm_map[cid] = None
            continue
        # excludes the synthetic probe/filler CRM record found during the schema audit
        if "filler" in d or d.get("call", {}).get("request_id") == "probe-0000":
            crm_map[cid] = None
        else:
            crm_map[cid] = d
    return crm_map, expected


def is_assembled(call_id, seed, ratio):
    h = hashlib.md5(f"{seed}:{call_id}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return frac < ratio


def word_metrics(whisper_text, soniox_text):
    w, s = whisper_text.split(), soniox_text.split()
    wc = len(w)
    if wc == 0:
        return 0.0, 0.0
    word_diff_pct = round(abs(wc - len(s)) / wc * 100, 2)
    cw, cs = Counter(w), Counter(s)
    overlap_pct = round(sum((cw & cs).values()) / wc * 100, 2)
    return word_diff_pct, overlap_pct


def main():
    args = parse_args()
    output_path = args.output_dir or f"asr_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    fs = HfFileSystem()
    crm_map, expected_segs = load_crm_and_expected(fs)

    files = list_shard_paths(fs)
    if args.max_shards:
        files = files[: args.max_shards]

    stats = Counter()
    stop = False

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing to {output_path}", file=sys.stderr)
    with open(output_path, "w", encoding="utf-8") as out_f:
        pbar = tqdm(files, desc="shards", unit="shard")
        for fp in pbar:
            if stop:
                break
            tbl = read_remote_table(fs, fp, SEG_COLS)
            rows = tbl.to_pylist()
            by_call = defaultdict(list)
            for r in rows:
                by_call[r["call_id"]].append(r)

            for call_id, segs in by_call.items():
                if args.max_calls and stats["calls_kept"] >= args.max_calls:
                    stop = True
                    break
                stats["calls_seen"] += 1

                usable = [s for s in segs if s["bucket"] != "unpaired" and s["text_whisper"]]
                if not usable:
                    stats["calls_skipped_no_usable"] += 1
                    continue

                assembled = is_assembled(call_id, args.seed, args.assembled_ratio)
                crm_context = crm_map.get(call_id)

                if assembled:
                    usable_by_channel = defaultdict(list)
                    for s in usable:
                        usable_by_channel[s["channel"]].append(s)

                    for channel, chan_segs in usable_by_channel.items():
                        # meta's n_seg_c0/n_seg_c1 reflects a different, pre-merge segmentation
                        # stage and is systematically off from the final row count here, so
                        # it's kept only as a non-blocking diagnostic, not a completeness gate.
                        expected = expected_segs.get(call_id, {}).get(channel)
                        seg_count_matches_meta = (expected is not None and len(chan_segs) == expected)
                        if not seg_count_matches_meta:
                            stats["calls_meta_segcount_mismatch"] += 1

                        chan_sorted = sorted(chan_segs, key=lambda s: s["start"])
                        whisper_full = " ".join(s["text_whisper"] for s in chan_sorted)
                        soniox_full = " ".join(s["text_raw"] for s in chan_sorted)
                        text_full = " ".join(s["text"] for s in chan_sorted)
                        turns = [
                            {
                                "start": s["start"], "end": s["end"],
                                "text_whisper": s["text_whisper"], "text_soniox": s["text_raw"],
                                "text": s["text"], "conf_mean": s["conf_mean"], "conf_min": s["conf_min"],
                                "wer_whisper": s["wer_whisper"], "bucket": s["bucket"],
                            }
                            for s in chan_sorted
                        ]
                        wd, ov = word_metrics(whisper_full, text_full)
                        conf_means = [s["conf_mean"] for s in chan_sorted if s["conf_mean"] is not None]
                        conf_mins = [s["conf_min"] for s in chan_sorted if s["conf_min"] is not None]
                        wers = [s["wer_whisper"] for s in chan_sorted if s["wer_whisper"] is not None]

                        row_out = {
                            "call_id": call_id,
                            "channel": channel,
                            "assembled": True,
                            "turns": turns,
                            "text_whisper": whisper_full,
                            "text_soniox": soniox_full,
                            "text": text_full,
                            "conf_mean": round(sum(conf_means) / len(conf_means), 4) if conf_means else None,
                            "conf_min": round(min(conf_mins), 4) if conf_mins else None,
                            "wer_whisper": round(sum(wers) / len(wers), 4) if wers else None,
                            "word_diff_pct": wd,
                            "overlap_pct": ov,
                            "crm_context": crm_context,
                            "seg_count_matches_meta": seg_count_matches_meta,
                        }
                        out_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                        stats["rows_assembled"] += 1
                else:
                    for s in usable:
                        wd, ov = word_metrics(s["text_whisper"], s["text"])
                        row_out = {
                            "call_id": call_id,
                            "channel": s["channel"],
                            "assembled": False,
                            "start": s["start"],
                            "end": s["end"],
                            "duration": s["duration"],
                            "text_whisper": s["text_whisper"],
                            "text_soniox": s["text_raw"],
                            "text": s["text"],
                            "conf_mean": s["conf_mean"],
                            "conf_min": s["conf_min"],
                            "wer_whisper": s["wer_whisper"],
                            "bucket": s["bucket"],
                            "word_diff_pct": wd,
                            "overlap_pct": ov,
                            "crm_context": None,
                        }
                        out_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                        stats["rows_chunked"] += 1
                stats["calls_kept"] += 1

            pbar.set_postfix(calls=stats["calls_kept"], rows=stats["rows_assembled"] + stats["rows_chunked"])

    print("=== STATS ===", file=sys.stderr)
    for k, v in stats.items():
        print(f"{k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
