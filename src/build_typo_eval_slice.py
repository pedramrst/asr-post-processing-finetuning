#!/usr/bin/env python3
"""Build a small typo/dictation-form eval slice from callcc-test-1k.

Complements build_entity_eval_slice.py: that one tracks named-entity
correction (mis-heard names/order numbers) separately from the aggregate
test_wer; this tracks the *other* main error category -- stutters,
word-boundary merges, ZWNJ half-spacing, letter-level slips -- which direct
investigation found the model already handles well. Tracking it as its own
slice (test_typo_wer) is a regression guard: as training leans harder on
entity correction (prepare_split.py's entity upsampling), this should catch
it if that quietly erodes the dictation-error correction that already works.

Classification reuses the same word-alignment approach as the entity slice,
but the opposite signal: for each reference word Whisper got wrong
(substitute-type only; deletions/insertions are structural, not "typo"-like),
compare it to Whisper's word with ZWNJ stripped from both. High character
overlap (shares most letters) means a spacing/stutter/boundary slip --
`--min-word-overlap` in the class of "میگیره" -> "می‌گیره" or "ههمینو" ->
"همینو". Low overlap means a different word entirely -- a genuine mishearing,
which belongs in the entity slice's territory, not this one. A row qualifies
if it has at least one typo-like error and *no* low-overlap (entity-like)
error, so the two slices stay disjoint for clean, separate tracking.

Example:
  python3 src/build_typo_eval_slice.py --output data/typo_eval_slice.jsonl
"""
import argparse
import json
import sys

import jiwer
from huggingface_hub import HfFileSystem


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="ErfanRou/callcc-test-1k")
    p.add_argument("--output", required=True, help="Path to write the typo eval slice .jsonl.")
    p.add_argument("--min-word-overlap", type=float, default=0.5,
                    help="Character-overlap fraction (of the reference word's own letters, ZWNJ-stripped) "
                         "above which a substitution counts as typo-like rather than a different word.")
    return p.parse_args()


def strip_zwnj(text: str) -> str:
    return text.replace("‌", "")


def char_overlap(a: str, b: str) -> float:
    """Fraction of `a`'s distinct characters also found in `b`."""
    a, b = strip_zwnj(a), strip_zwnj(b)
    if not a:
        return 0.0
    return len(set(a) & set(b)) / len(set(a))


def classify_row(text_whisper: str, text: str, min_overlap: float) -> str:
    """Returns "typo", "entity", "both", or "neither" for one row's substitution errors."""
    out = jiwer.process_words([text], [text_whisper])
    ref_words = text.split()
    has_typo = has_entity = False
    for chunk in out.alignments[0]:
        if chunk.type != "substitute":
            continue
        for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
            hyp_words = text_whisper.split()[chunk.hyp_start_idx:chunk.hyp_end_idx]
            hyp_word = " ".join(hyp_words)
            if not hyp_word:
                continue
            if char_overlap(ref_words[i], hyp_word) >= min_overlap:
                has_typo = True
            else:
                has_entity = True
    if has_typo and has_entity:
        return "both"
    if has_typo:
        return "typo"
    if has_entity:
        return "entity"
    return "neither"


def main():
    args = parse_args()
    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{args.repo_id}/data/*.parquet"))
    if not paths:
        print(f"No shards found for {args.repo_id}", file=sys.stderr)
        sys.exit(1)

    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["call_id", "channel", "text_whisper", "text", "text_raw"]
    tables = []
    for path in paths:
        with fs.open(path, "rb") as f:
            tables.append(pq.ParquetFile(f).read(columns=cols, use_threads=True))
    rows = pa.concat_tables(tables).to_pylist()

    out_rows = []
    label_counts = {"typo": 0, "entity": 0, "both": 0, "neither": 0}
    for r in rows:
        if not r["text_whisper"] or not r["text"]:
            continue
        label = classify_row(r["text_whisper"], r["text"], args.min_word_overlap)
        label_counts[label] += 1
        if label == "typo":
            out_rows.append({
                "call_id": r["call_id"],
                "channel": r["channel"],
                "text_whisper": r["text_whisper"],
                "text": r["text"],
                # Punctuated version of `text` -- see Config.include_punctuation.
                "text_raw": r["text_raw"],
            })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"scanned {len(rows)} rows from {args.repo_id}", file=sys.stderr)
    print(f"label counts: {label_counts}", file=sys.stderr)
    print(f"wrote {len(out_rows)} typo-only rows (no entity-like error present) to {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
