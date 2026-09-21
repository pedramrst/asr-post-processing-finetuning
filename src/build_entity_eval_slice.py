#!/usr/bin/env python3
"""Build a small, ground-truth-backed named-entity eval slice from callcc-test-1k.

Aggregate WER on the full test set mixes entity-heavy and entity-free calls
together, which hides exactly the failure mode this task's system prompt
cares about most (person/place/order-name correction) inside a number
dominated by everything else. callcc-test-1k carries `crm_metadata` (the real
CRM record for that call) on every row -- this cross-references the
customer's real name and the CRM record's `owner_name` (usually the agent's
own name, in Persian) against the reference transcript, and keeps only rows
where at least one confirmed real name-word actually appears in it. That
gives a slice you know for certain involves a genuine named entity, not a
heuristic guess.

`entity_word_missing_from_whisper` flags whether Whisper's transcription
already differs from the reference on that confirmed name word -- useful for
filtering to just the "genuinely wrong" subset later, but the base slice
intentionally includes rows where Whisper got it right too, since correction
quality on entities generally (not just error-recovery) is the point.

Only two rare/short-name filters are applied: a matched word must be >= 3
characters (excludes single-letter/2-char tokens too generic to be a reliable
name signal) and must appear as a whole word in the reference (not a
substring), to avoid false positives from common words that happen to
contain a name as a substring.

Example:
  python3 src/build_entity_eval_slice.py --output data/entity_eval_slice.jsonl
"""
import argparse
import json
import re
import sys

from huggingface_hub import HfFileSystem


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="ErfanRou/callcc-test-1k")
    p.add_argument("--output", required=True, help="Path to write the entity eval slice .jsonl.")
    p.add_argument("--min-name-word-len", type=int, default=3,
                    help="Minimum character length for a CRM name word to count as a match.")
    return p.parse_args()


def persian_name_words(crm: dict, min_len: int) -> set[str]:
    """Real name words from this call's CRM record: the customer's name and
    the CRM case owner's name (usually the agent, in Persian -- unlike
    crm.agent.name, which is in Latin script and won't match transcript text)."""
    words: set[str] = set()
    customer_name = (crm.get("customer", {}) or {}).get("name") or ""
    owner_name = (crm.get("crm", {}) or {}).get("owner_name") or ""
    owner_name = re.sub(r"\(.*?\)", "", owner_name)  # strip "(فرانت)"-style role suffixes
    for source in (customer_name, owner_name):
        for word in source.split():
            if len(word) >= min_len:
                words.add(word)
    return words


def main():
    args = parse_args()
    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{args.repo_id}/data/*.parquet"))
    if not paths:
        print(f"No shards found for {args.repo_id}", file=sys.stderr)
        sys.exit(1)

    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["call_id", "channel", "text_whisper", "text", "text_raw", "crm_metadata"]
    tables = []
    for path in paths:
        with fs.open(path, "rb") as f:
            tables.append(pq.ParquetFile(f).read(columns=cols, use_threads=True))
    rows = pa.concat_tables(tables).to_pylist()

    out_rows = []
    for r in rows:
        if not r["text_whisper"] or not r["text"] or not r["crm_metadata"]:
            continue
        crm = json.loads(r["crm_metadata"])
        name_words = persian_name_words(crm, args.min_name_word_len)
        if not name_words:
            continue
        ref_words = r["text"].split()
        hit_words = sorted({w for w in ref_words if w in name_words})
        if not hit_words:
            continue
        whisper_words = set(r["text_whisper"].split())
        out_rows.append({
            "call_id": r["call_id"],
            "channel": r["channel"],
            "text_whisper": r["text_whisper"],
            "text": r["text"],
            # Punctuated version of `text` -- see Config.include_punctuation.
            "text_raw": r["text_raw"],
            "entity_words": hit_words,
            "entity_word_missing_from_whisper": any(w not in whisper_words for w in hit_words),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_mismatch = sum(1 for r in out_rows if r["entity_word_missing_from_whisper"])
    print(f"scanned {len(rows)} rows from {args.repo_id}", file=sys.stderr)
    print(f"wrote {len(out_rows)} rows with a confirmed entity word to {args.output}", file=sys.stderr)
    print(f"{n_mismatch}/{len(out_rows)} of those have Whisper actually wrong on that word", file=sys.stderr)


if __name__ == "__main__":
    main()
