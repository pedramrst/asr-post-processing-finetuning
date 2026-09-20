"""Curate build_dataset.py's raw output into a training-ready split.

Applies three fixes, based on inspecting an actual sample of the raw data:

  1. Drops pairs where wer_whisper is implausibly high. The highest-WER raw
     examples (1.6-4.0) turned out to be segment misalignments -- Whisper's
     window drifted onto a different utterance entirely -- not genuine
     correctable ASR errors. Training on them would teach the model to
     hallucinate unrelated rewrites rather than correct real mistakes.
  2. Upsamples low-WER ("agree"-bucket) chunked rows. Natural rate is only
     ~11%; left alone, the model rarely sees an already-correct transcript
     and may learn to always rewrite, over-correcting in production.
  3. Caps the share of assembled (full-channel) rows. They give useful
     cross-segment context but are much longer and shouldn't dominate the
     primarily segment-level training signal.

Only chunked rows have a `bucket` label, so the upsampling step only looks
at those; the assembled cap applies across the whole output.

Example:
  python3 src/prepare_split.py --input asr_dataset.jsonl --output asr_dataset_curated.jsonl
"""
import argparse
import json
import random
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Raw .jsonl from build_dataset.py.")
    p.add_argument("--output", required=True, help="Path to write the curated .jsonl.")
    p.add_argument("--wer-cap", type=float, default=1.0,
                    help="Drop rows with wer_whisper above this (misaligned pairs, not real correction cases).")
    p.add_argument("--agree-target-frac", type=float, default=0.225,
                    help="Target share of low-WER ('agree'-bucket) rows among chunked rows after upsampling.")
    p.add_argument("--assembled-target-frac", type=float, default=0.10,
                    help="Target share of assembled rows in the final output (only ever downsampled, never inflated).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def resample(rows, target_count, rng):
    """Sample `rows` up or down to exactly target_count, deterministically."""
    if not rows or target_count <= len(rows):
        return rng.sample(rows, target_count)
    extra = target_count - len(rows)
    return list(rows) + rng.choices(rows, k=extra)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = load_rows(args.input)

    kept = [r for r in rows if r.get("wer_whisper") is not None and r["wer_whisper"] <= args.wer_cap]
    dropped = len(rows) - len(kept)

    chunked = [r for r in kept if not r["assembled"]]
    assembled = [r for r in kept if r["assembled"]]

    agree = [r for r in chunked if r.get("bucket") == "agree"]
    other = [r for r in chunked if r.get("bucket") != "agree"]

    frac = args.agree_target_frac
    agree_target = round(frac / (1 - frac) * len(other)) if frac < 1 else len(agree)
    agree_final = resample(agree, agree_target, rng) if agree else []

    chunked_final = other + agree_final

    a_frac = args.assembled_target_frac
    assembled_target = round(a_frac / (1 - a_frac) * len(chunked_final)) if a_frac < 1 else len(assembled)
    assembled_final = (
        rng.sample(assembled, assembled_target) if assembled_target < len(assembled) else assembled
    )

    final_rows = chunked_final + assembled_final
    rng.shuffle(final_rows)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"input rows: {len(rows)}", file=sys.stderr)
    print(f"dropped (wer_whisper > {args.wer_cap} or missing): {dropped}", file=sys.stderr)
    print(f"chunked other-bucket rows kept: {len(other)}", file=sys.stderr)
    print(f"agree-bucket rows: {len(agree)} -> upsampled to {len(agree_final)} "
          f"(target frac {frac})", file=sys.stderr)
    print(f"assembled rows: {len(assembled)} -> kept {len(assembled_final)} "
          f"(target frac {a_frac})", file=sys.stderr)
    print(f"final rows written to {args.output}: {len(final_rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
