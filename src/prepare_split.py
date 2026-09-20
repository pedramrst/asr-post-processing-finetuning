"""Curate build_dataset.py's raw output into a training-ready split.

Applies three fixes, based on inspecting an actual sample of the raw data:

  1. Drops pairs where wer_whisper is implausibly high, or word-overlap
     between text_whisper and the target is implausibly low. The highest-WER
     raw examples (1.6-4.0) turned out to be segment misalignments --
     Whisper's window drifted onto a different utterance entirely -- not
     genuine correctable ASR errors. Training on them would teach the model
     to hallucinate unrelated rewrites rather than correct real mistakes.
     overlap_pct is a second, largely independent signal for the same
     failure mode -- WER is length-normalized edit distance and can stay
     misleadingly low on a misaligned pair, while overlap_pct (share of
     text_whisper's words that appear anywhere in the target) collapses
     toward zero whenever the two sides are actually about different
     content.
  2. Upsamples low-WER ("agree"-bucket) chunked rows. Natural rate is only
     ~11%; left alone, the model rarely sees an already-correct transcript
     and may learn to always rewrite, over-correcting in production.
  3. Balances assembled (full-channel) vs. chunked (per-segment) rows to
     --assembled-target-frac. Production runs correction over one full call
     channel at a time, so assembled rows are the primary training signal,
     not a minority -- chunked rows are kept as a smaller supplementary set
     for the low-WER "leave it alone" signal above (only chunked rows get a
     `bucket` label) and segment-level diversity. Whichever side is
     oversupplied relative to the target ratio gets downsampled; neither
     side is ever upsampled to hit it (so an insufficient supply of
     assembled rows here means going back to build_dataset.py's
     --assembled-ratio, not raising this flag further).

Only chunked rows have a `bucket` label, so the upsampling step only looks
at those; the assembled/chunked balance step applies across the whole output.

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
    p.add_argument("--overlap-floor", type=float, default=30.0,
                    help="Drop rows with overlap_pct (word-overlap between text_whisper and the target) "
                         "below this -- catches misaligned pairs wer_cap's length-normalized WER can miss. "
                         "Starting point, not tuned against real data -- check the drop count printed below "
                         "and adjust.")
    p.add_argument("--agree-target-frac", type=float, default=0.225,
                    help="Target share of low-WER ('agree'-bucket) rows among chunked rows after upsampling.")
    p.add_argument("--assembled-target-frac", type=float, default=0.85,
                    help="Target share of assembled (full-channel) rows in the final output -- "
                         "the primary training signal, since production corrects one full call "
                         "channel at a time. Whichever of assembled/chunked is oversupplied "
                         "relative to this ratio gets downsampled; neither side is ever upsampled.")
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


def downsample_to_ratio(rows_a, rows_b, target_frac_a, rng):
    """Downsample whichever of `rows_a`/`rows_b` is oversupplied so `rows_a`
    ends up at `target_frac_a` of the combined total.

    Never upsamples either side: if `rows_a` is too scarce to reach the
    ratio by only trimming `rows_b`, `rows_b` is downsampled to match `rows_a`
    instead of inflating it. This is the reason a target fraction can end up
    unreachable from a given raw mix -- fix the supply upstream (e.g.
    build_dataset.py's --assembled-ratio) rather than raising this further.
    """
    if target_frac_a <= 0:
        return [], rows_b
    if target_frac_a >= 1:
        return rows_a, []
    n_a, n_b = len(rows_a), len(rows_b)
    if n_a == 0 or n_b == 0:
        return rows_a, rows_b
    target_a = round(target_frac_a / (1 - target_frac_a) * n_b)
    if target_a <= n_a:
        return rng.sample(rows_a, target_a), rows_b
    target_b = round((1 - target_frac_a) / target_frac_a * n_a)
    return rows_a, rng.sample(rows_b, min(target_b, n_b))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = load_rows(args.input)

    wer_ok = [r for r in rows if r.get("wer_whisper") is not None and r["wer_whisper"] <= args.wer_cap]
    kept = [r for r in wer_ok if r.get("overlap_pct") is not None and r["overlap_pct"] >= args.overlap_floor]
    dropped_wer = len(rows) - len(wer_ok)
    dropped_overlap = len(wer_ok) - len(kept)

    chunked = [r for r in kept if not r["assembled"]]
    assembled = [r for r in kept if r["assembled"]]

    agree = [r for r in chunked if r.get("bucket") == "agree"]
    other = [r for r in chunked if r.get("bucket") != "agree"]

    frac = args.agree_target_frac
    agree_target = round(frac / (1 - frac) * len(other)) if frac < 1 else len(agree)
    agree_final = resample(agree, agree_target, rng) if agree else []

    chunked_before_balance = other + agree_final

    assembled_final, chunked_final = downsample_to_ratio(
        assembled, chunked_before_balance, args.assembled_target_frac, rng
    )

    final_rows = chunked_final + assembled_final
    rng.shuffle(final_rows)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    achieved_frac = len(assembled_final) / len(final_rows) if final_rows else 0.0
    print(f"input rows: {len(rows)}", file=sys.stderr)
    print(f"dropped (wer_whisper > {args.wer_cap} or missing): {dropped_wer}", file=sys.stderr)
    print(f"dropped (overlap_pct < {args.overlap_floor} or missing, among wer-ok rows): {dropped_overlap}",
          file=sys.stderr)
    print(f"chunked other-bucket rows kept: {len(other)}", file=sys.stderr)
    print(f"agree-bucket rows: {len(agree)} -> upsampled to {len(agree_final)} "
          f"(target frac {frac})", file=sys.stderr)
    print(f"assembled rows: {len(assembled)} -> kept {len(assembled_final)}; "
          f"chunked rows: {len(chunked_before_balance)} -> kept {len(chunked_final)} "
          f"(target assembled frac {args.assembled_target_frac}, achieved {achieved_frac:.3f})",
          file=sys.stderr)
    print(f"final rows written to {args.output}: {len(final_rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
