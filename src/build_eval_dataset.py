#!/usr/bin/env python3
"""Combine the entity and typo eval slices into one eval dataset.

build_entity_eval_slice.py and build_typo_eval_slice.py each produce a
standalone .jsonl; this stacks both into one file/Hub repo instead of two,
tagged with a `row_type` column ("entity"/"typo") so a single dataset_id can
serve both of Config's test_entity_dataset_id/test_typo_dataset_id --
train.py filters back down to one slice at eval time via
test_entity_row_type/test_typo_row_type (see config.py), so this is
functionally identical to pushing two separate repos, just one repo to
manage instead of two.

Only the columns both slices actually share (plus text_raw, used when
Config.include_punctuation) are kept -- the entity slice's extra metadata
(entity_words/entity_spans/entity_sources/entity_word_missing_from_whisper)
doesn't have a typo-slice equivalent and isn't read by the eval loading
path anyway (see evaluate.py's run_test_eval, which only ever pulls
input_column/target_column/row_type), so dropping it here keeps the combined
schema simple instead of padding typo rows with null entity fields.

Example:
  python3 src/build_eval_dataset.py \
      --entity-input data/entity_eval_slice.jsonl \
      --typo-input data/typo_eval_slice.jsonl \
      --output data/eval_dataset.jsonl
"""
import argparse
import json
import sys

_SHARED_COLUMNS = ["call_id", "channel", "text_whisper", "text", "text_raw"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity-input", required=True, help="build_entity_eval_slice.py's output .jsonl.")
    p.add_argument("--typo-input", required=True, help="build_typo_eval_slice.py's output .jsonl.")
    p.add_argument("--output", required=True, help="Path to write the combined .jsonl.")
    return p.parse_args()


def _load_tagged(path: str, row_type: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = {c: obj.get(c) for c in _SHARED_COLUMNS}
            row["row_type"] = row_type
            rows.append(row)
    return rows


def main():
    args = parse_args()
    entity_rows = _load_tagged(args.entity_input, "entity")
    typo_rows = _load_tagged(args.typo_input, "typo")
    if not entity_rows or not typo_rows:
        print("One of --entity-input/--typo-input produced zero rows -- check the paths.", file=sys.stderr)
        sys.exit(1)

    combined = entity_rows + typo_rows
    with open(args.output, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(combined)} rows to {args.output} "
          f"({len(entity_rows)} entity, {len(typo_rows)} typo)", file=sys.stderr)


if __name__ == "__main__":
    main()
