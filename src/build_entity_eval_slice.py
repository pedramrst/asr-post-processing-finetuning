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

Optionally, pass --entities-dir to widen the slice beyond CRM-confirmed
personal names using a second, independent source: a directory of per-call
LLM (Gemini) named-entity extractions keyed by call_id (see
data/output-backup/*.json -- span/category/subtype/confidence per entity,
run over the same callcc-test-1k reference transcripts). Each entity's `span`
is matched against `text` the same way CRM name-words are (whole word/phrase,
not a substring), so a row is kept if *either* source confirms an entity.
This is LLM-labeled, not human-verified, so treat it as a second signal
rather than ground truth -- `entity_sources` on each row says whether a hit
came from "crm", "gemini", or both, so you can filter to the stricter
CRM-only rows later if needed.

Example:
  python3 src/build_entity_eval_slice.py --output data/entity_eval_slice.jsonl
  python3 src/build_entity_eval_slice.py --output data/entity_eval_slice.jsonl \
      --entities-dir data/output-backup
"""
import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfFileSystem


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="ErfanRou/callcc-test-1k")
    p.add_argument("--output", required=True, help="Path to write the entity eval slice .jsonl.")
    p.add_argument("--min-name-word-len", type=int, default=3,
                    help="Minimum character length for a CRM name word, or a Gemini entity span, to count as a "
                         "match.")
    p.add_argument("--entities-dir", default=None,
                    help="Optional dir of per-call Gemini entity-extraction JSON files (keyed by call_id) to "
                         "widen the slice beyond CRM name matches. See module docstring.")
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


def load_gemini_entities(entities_dir: str) -> dict[str, list[dict]]:
    """call_id -> pooled entity dicts from a teammate's Gemini extraction backup
    (data/output-backup/*.json), flattened across that file's `groups`."""
    entities_by_call: dict[str, list[dict]] = {}
    for path in Path(entities_dir).glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        call_id = d.get("call_id") or path.stem
        ents = [e for g in d.get("groups", {}).values() for e in g.get("entities", [])]
        entities_by_call[call_id] = ents
    return entities_by_call


def whole_span_in_text(span: str, text: str) -> bool:
    """True if `span` (one or more words) appears in `text` bounded by
    whitespace on both sides -- i.e. as itself, not as part of a larger word."""
    return re.search(r"(?<!\S)" + re.escape(span) + r"(?!\S)", text) is not None


def gemini_span_hits(text: str, entities: list[dict], min_len: int) -> list[dict]:
    """Entities whose `span` is long enough and actually appears in `text`,
    same match rule as persian_name_words but extended to multi-word spans."""
    return [e for e in entities
            if len((e.get("span") or "").strip()) >= min_len and whole_span_in_text(e["span"].strip(), text)]


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

    gemini_entities_by_call = load_gemini_entities(args.entities_dir) if args.entities_dir else {}

    out_rows = []
    for r in rows:
        if not r["text_whisper"] or not r["text"]:
            continue

        crm = json.loads(r["crm_metadata"]) if r["crm_metadata"] else {}
        name_words = persian_name_words(crm, args.min_name_word_len)
        crm_hits = sorted({w for w in r["text"].split() if w in name_words})

        gemini_hits = gemini_span_hits(
            r["text"], gemini_entities_by_call.get(r["call_id"], []), args.min_name_word_len)

        entity_words = sorted(set(crm_hits) | {e["span"].strip() for e in gemini_hits})
        if not entity_words:
            continue

        sources = [s for s, hit in (("crm", crm_hits), ("gemini", gemini_hits)) if hit]
        out_rows.append({
            "call_id": r["call_id"],
            "channel": r["channel"],
            "text_whisper": r["text_whisper"],
            "text": r["text"],
            # Punctuated version of `text` -- see Config.include_punctuation.
            "text_raw": r["text_raw"],
            "entity_words": entity_words,
            "entity_spans": [{"span": e["span"].strip(), "category": e.get("category"),
                               "subtype": e.get("subtype")} for e in gemini_hits],
            "entity_sources": sources,
            "entity_word_missing_from_whisper": any(
                not whole_span_in_text(w, r["text_whisper"]) for w in entity_words),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_mismatch = sum(1 for r in out_rows if r["entity_word_missing_from_whisper"])
    n_crm_only = sum(1 for r in out_rows if r["entity_sources"] == ["crm"])
    n_gemini_only = sum(1 for r in out_rows if r["entity_sources"] == ["gemini"])
    n_both = sum(1 for r in out_rows if len(r["entity_sources"]) == 2)
    print(f"scanned {len(rows)} rows from {args.repo_id}", file=sys.stderr)
    print(f"wrote {len(out_rows)} rows with a confirmed entity word to {args.output}", file=sys.stderr)
    if args.entities_dir:
        print(f"  by source: crm-only {n_crm_only}, gemini-only {n_gemini_only}, both {n_both}", file=sys.stderr)
    print(f"{n_mismatch}/{len(out_rows)} of those have Whisper actually wrong on that word", file=sys.stderr)


if __name__ == "__main__":
    main()
