"""Curate build_dataset.py's raw output into a training-ready split.

Applies four fixes, based on inspecting an actual sample of the raw data:

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
  4. Upsamples rows containing a confirmed named entity. Each real name/
     order-number appears in maybe one or two calls total, so without this
     the model sees very few gradient updates per entity relative to common
     dictation errors that repeat across hundreds of rows -- directly
     investigating fine-tuned predictions found the model reliably fixes
     dictation-form errors (stutters, word-boundary merges, ZWNJ) but had
     zero confirmed successes on an actually mis-heard name in a small
     sample, despite a much higher hit rate on generic rare words. Detection
     reuses build_dataset.py's `crm_context` (the real CRM record for that
     call, only ever present on assembled rows) the same way
     build_entity_eval_slice.py does for the eval slice: a row counts as
     "entity" if the CRM customer name or case-owner name has a word that
     actually appears in the target text -- i.e. a confirmed real entity,
     not a rare-word heuristic guess. --entity-max-repeats caps how many
     times any single row can be duplicated to get there: distinct customer
     names scale roughly linearly with how many raw calls you process (real
     measurement: ~324 distinct names per 6,000 calls), but --entity-target-
     frac is a fraction of the whole output regardless of corpus size, so
     without a cap a small distinct pool relative to a large non-entity pool
     still gets repeated dozens of times each -- teaching the model to
     over-memorize a handful of specific people rather than the general
     skill. Process more raw calls (a higher build_dataset.py --max-calls,
     or none at all) for a larger, more diverse distinct-entity pool instead
     of raising the cap.

Only chunked rows have a `bucket` label, so the upsampling step only looks
at those; the assembled/chunked balance and entity-upsampling steps apply
across the whole output.

Example:
  python3 src/prepare_split.py --input asr_dataset.jsonl --output asr_dataset_curated.jsonl
"""
import argparse
import json
import random
import re
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
    p.add_argument("--entity-target-frac", type=float, default=0.5,
                    help="Target share of confirmed-named-entity rows (see build_entity_eval_slice.py's "
                         "detection method) in the final output. Only assembled rows ever carry crm_context, "
                         "so this only ever finds/upsamples within that pool. 0 disables it.")
    p.add_argument("--entity-min-name-word-len", type=int, default=3,
                    help="Minimum character length for a CRM name word to count as a match.")
    p.add_argument("--entity-max-repeats", type=int, default=5,
                    help="Cap on how many times any single confirmed-entity row can be duplicated when "
                         "upsampling -- without this, a small distinct-entity pool relative to "
                         "--entity-target-frac gets repeated dozens of times each, over-memorizing a handful "
                         "of specific people instead of teaching the general correction skill. When the cap "
                         "makes the target unreachable, the achieved fraction (printed below) falls short of "
                         "it rather than over-duplicating to compensate.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def persian_name_words(crm: dict, min_len: int) -> set[str]:
    """Real name words from a call's CRM record: the customer's name and the
    CRM case owner's name (usually the agent, in Persian). Mirrors
    build_entity_eval_slice.py's detection exactly, so training upsampling
    and the held-out entity eval slice agree on what counts as an entity.
    """
    words: set[str] = set()
    customer_name = (crm.get("customer", {}) or {}).get("name") or ""
    owner_name = (crm.get("crm", {}) or {}).get("owner_name") or ""
    owner_name = re.sub(r"\(.*?\)", "", owner_name)
    for source in (customer_name, owner_name):
        for word in source.split():
            if len(word) >= min_len:
                words.add(word)
    return words


def is_entity_row(row: dict, min_len: int) -> bool:
    crm = row.get("crm_context")
    if not crm:
        return False
    name_words = persian_name_words(crm, min_len)
    if not name_words:
        return False
    return bool(name_words & set(row["text"].split()))


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def resample(rows, target_count, rng):
    """Sample `rows` up or down to exactly target_count, deterministically."""
    if not rows or target_count <= len(rows):
        return rng.sample(rows, target_count)
    extra = target_count - len(rows)
    return list(rows) + rng.choices(rows, k=extra)


def resample_capped(rows, target_count, max_repeats, rng):
    """Like resample(), but never lets any single row appear more than
    `max_repeats` times when upsampling (downsampling is unaffected).

    A distinct population that's small relative to `target_count` -- as a
    confirmed-entity pool typically is, since most calls don't have one --
    would otherwise get duplicated dozens of times each to hit a target
    fraction, which teaches the model to over-memorize a handful of specific
    people rather than the general skill. When the cap makes `target_count`
    unreachable, this returns fewer rows than asked and the caller's
    printed "achieved" fraction reports the shortfall honestly rather than
    silently over-duplicating to hit the nominal target.
    """
    if not rows or target_count <= len(rows):
        return rng.sample(rows, target_count)
    target_count = min(target_count, len(rows) * max_repeats)
    extra = target_count - len(rows)
    pool = []
    while len(pool) < extra:
        cycle = list(rows)
        rng.shuffle(cycle)
        pool.extend(cycle)
    return list(rows) + pool[:extra]


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

    # Entity upsampling runs last, over the whole post-balance pool -- only
    # assembled rows can ever match (chunked rows never carry crm_context),
    # so this can't disturb the assembled/chunked ratio just established
    # above by pulling additional rows from the chunked side.
    entity_rows, non_entity_rows = [], []
    for r in chunked_final + assembled_final:
        (entity_rows if is_entity_row(r, args.entity_min_name_word_len) else non_entity_rows).append(r)

    e_frac = args.entity_target_frac
    if e_frac <= 0 or not entity_rows:
        entity_final = entity_rows  # disabled, or nothing to upsample -- leave the natural rate as-is
    else:
        entity_target = round(e_frac / (1 - e_frac) * len(non_entity_rows)) if e_frac < 1 else len(entity_rows)
        entity_final = resample_capped(entity_rows, entity_target, args.entity_max_repeats, rng)

    final_rows = non_entity_rows + entity_final
    rng.shuffle(final_rows)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Computed from final_rows directly (not assembled_final/chunked_final's
    # pre-entity-upsampling counts), since entity upsampling only ever adds
    # assembled-origin duplicates and would otherwise silently shift the
    # true final ratio away from what those counts suggest.
    achieved_frac = sum(1 for r in final_rows if r["assembled"]) / len(final_rows) if final_rows else 0.0
    print(f"input rows: {len(rows)}", file=sys.stderr)
    print(f"dropped (wer_whisper > {args.wer_cap} or missing): {dropped_wer}", file=sys.stderr)
    print(f"dropped (overlap_pct < {args.overlap_floor} or missing, among wer-ok rows): {dropped_overlap}",
          file=sys.stderr)
    print(f"chunked other-bucket rows kept: {len(other)}", file=sys.stderr)
    print(f"agree-bucket rows: {len(agree)} -> upsampled to {len(agree_final)} "
          f"(target frac {frac})", file=sys.stderr)
    pre_entity_total = len(assembled_final) + len(chunked_final)
    pre_entity_frac = len(assembled_final) / pre_entity_total if pre_entity_total else 0.0
    print(f"assembled rows: {len(assembled)} -> kept {len(assembled_final)}; "
          f"chunked rows: {len(chunked_before_balance)} -> kept {len(chunked_final)} "
          f"(target assembled frac {args.assembled_target_frac}, "
          f"achieved before entity upsampling {pre_entity_frac:.3f}, "
          f"achieved in final output {achieved_frac:.3f})",
          file=sys.stderr)
    entity_achieved = len(entity_final) / len(final_rows) if final_rows else 0.0
    avg_repeats = len(entity_final) / len(entity_rows) if entity_rows else 0.0
    capped_note = " (--entity-max-repeats limited this short of the target)" if len(entity_final) < (
        round(e_frac / (1 - e_frac) * len(non_entity_rows)) if 0 < e_frac < 1 else len(entity_rows)
    ) else ""
    print(f"confirmed-entity rows: {len(entity_rows)} -> upsampled to {len(entity_final)} "
          f"(target frac {e_frac}, achieved {entity_achieved:.3f}, avg {avg_repeats:.1f}x repeats per row, "
          f"cap {args.entity_max_repeats}x){capped_note}", file=sys.stderr)
    print(f"final rows written to {args.output}: {len(final_rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
