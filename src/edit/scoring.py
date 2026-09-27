"""Scores corrected transcripts for the edit-based method.

The core numbers (wer, wer_zwnj_normalized, exact_match, fix_rate,
preservation_rate, their *_lenient variants, fix_rate_recoverable,
hallucination_rate) are computed exactly the way evaluate.py's
run_test_eval() computes them, reusing its alignment helpers, so a result
here is directly comparable with the rewrite method's Hub metrics.json
files. score() on a rewrite run's own predictions reproduces its
metrics.json -- diff_filter_baseline.py checks that before trusting
anything else it reports.

On top of those, the metrics the edit method is actually judged on
(README "Evaluation" in this folder):

  * lenient fixed / broken word counts and their difference (net_fixed):
    a rate can't show that fixing 30 words while breaking 300 is a loss.
  * broken_per_100_correct: new errors per 100 words Whisper already had
    right -- the "false edits on clean text" number.
  * rows_changed: fraction of transcripts touched at all.
  * entity metrics against reference entity spans (CRM names + Gemini
    extractions, see grounding.reference_entity_spans): of the entity spans
    Whisper got wrong, how many the correction recovered (entity_fix_rate);
    of those Whisper got right, how many survived (entity_keep_rate).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

import jiwer  # noqa: E402

from data import low_signal_word_ranges  # noqa: E402
from evaluate import (  # noqa: E402
    _equal_or_equivalent_positions,
    _equal_positions,
    _strip_zwnj,
    _word_overlap_pct,
)
from persian_normalize import normalize_lenient  # noqa: E402

HALLUCINATION_OVERLAP_FLOOR = 50.0  # evaluate.py's default
FIX_WEIGHT = 0.5  # evaluate.py's default


def _contains_span(text: str, span: str) -> bool:
    """Whole-word(s) containment, ZWNJ/spacing-insensitive (so "دیجی‌کالا"
    and "دیجی کالا" both count as mentioning the entity)."""
    words = normalize_lenient(text).split()
    span_words = normalize_lenient(span).split()
    k = len(span_words)
    return k > 0 and any(words[i:i + k] == span_words for i in range(len(words) - k + 1))


def lenient_positions(source: str, hypothesis: str, reference: str) -> tuple[set[int], set[int], set[int], str]:
    """(target, already_correct, still_correct) lenient reference-word
    position sets, plus the normalized reference -- evaluate.py's lenient
    bucket split, factored out so per-edit analysis can reuse it."""
    ref_l = normalize_lenient(reference)
    src_l = normalize_lenient(source)
    hyp_l = normalize_lenient(hypothesis)
    ref_w = ref_l.split()
    already = _equal_or_equivalent_positions(
        jiwer.process_words([ref_l], [src_l]).alignments[0], ref_w, src_l.split())
    still = _equal_or_equivalent_positions(
        jiwer.process_words([ref_l], [hyp_l]).alignments[0], ref_w, hyp_l.split()) if hyp_l else set()
    target = set(range(len(ref_w))) - already
    return target, already, still, ref_l


def score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    corpus_freq: Counter | None = None,
    entity_spans: list[list[str]] | None = None,
    low_signal_min_similarity: float = 0.75,
    low_signal_max_common_freq: int = 1,
) -> dict:
    """Corpus-level metrics for `hypotheses` (corrections of `sources`)
    against `references`. `entity_spans[i]` is the list of reference entity
    strings for row i (optional)."""
    n = len(references)
    m: dict = {"n_examples": n}
    m["wer"] = jiwer.process_words(references, hypotheses).wer
    m["wer_zwnj_normalized"] = jiwer.process_words(
        [_strip_zwnj(r) for r in references], [_strip_zwnj(h) for h in hypotheses]).wer
    m["exact_match"] = sum(h.strip() == r.strip() for h, r in zip(hypotheses, references)) / n
    m["hallucination_rate"] = sum(
        _word_overlap_pct(h, s) < HALLUCINATION_OVERLAP_FLOOR for h, s in zip(hypotheses, sources)) / n
    m["rows_changed"] = sum(h.split() != s.split() for h, s in zip(hypotheses, sources)) / n

    fix_h = fix_t = keep_h = keep_t = 0
    fix_hl = fix_tl = keep_hl = keep_tl = 0
    fix_hr = fix_tr = unrecoverable = 0
    ent_fix_h = ent_fix_t = ent_keep_h = ent_keep_t = 0
    for i, (src, hyp, ref) in enumerate(zip(sources, hypotheses, references)):
        # Strict (byte-exact) buckets -- evaluate.py's fix_rate/preservation_rate.
        already = _equal_positions(jiwer.process_words([ref], [src]).alignments[0])
        still = _equal_positions(jiwer.process_words([ref], [hyp]).alignments[0]) if hyp.strip() else set()
        target = set(range(len(ref.split()))) - already
        fix_h += len(target & still)
        fix_t += len(target)
        keep_h += len(already & still)
        keep_t += len(already)

        target_l, already_l, still_l, ref_l = lenient_positions(src, hyp, ref)
        fix_hl += len(target_l & still_l)
        fix_tl += len(target_l)
        keep_hl += len(already_l & still_l)
        keep_tl += len(already_l)

        if corpus_freq is not None:
            bad = set()
            for first, last in low_signal_word_ranges(
                normalize_lenient(src), ref_l, low_signal_min_similarity, corpus_freq, low_signal_max_common_freq,
            ):
                bad.update(range(first, last + 1))
            target_r = target_l - bad
            fix_hr += len(target_r & still_l)
            fix_tr += len(target_r)
            unrecoverable += len(target_l & bad)

        if entity_spans is not None:
            for span in entity_spans[i]:
                hit = _contains_span(hyp, span)
                if _contains_span(src, span):
                    ent_keep_t += 1
                    ent_keep_h += hit
                else:
                    ent_fix_t += 1
                    ent_fix_h += hit

    def ratio(a, b):
        return a / b if b else None

    m["fix_rate"] = ratio(fix_h, fix_t)
    m["preservation_rate"] = ratio(keep_h, keep_t)
    m["targeted_score"] = FIX_WEIGHT * m["fix_rate"] + (1 - FIX_WEIGHT) * m["preservation_rate"]
    m["fix_rate_lenient"] = ratio(fix_hl, fix_tl)
    m["preservation_rate_lenient"] = ratio(keep_hl, keep_tl)
    m["targeted_score_lenient"] = (
        FIX_WEIGHT * m["fix_rate_lenient"] + (1 - FIX_WEIGHT) * m["preservation_rate_lenient"])
    m["fixed_lenient"] = fix_hl
    m["broken_lenient"] = keep_tl - keep_hl
    m["net_fixed_lenient"] = fix_hl - (keep_tl - keep_hl)
    m["broken_per_100_correct"] = 100 * (keep_tl - keep_hl) / keep_tl if keep_tl else None
    if corpus_freq is not None:
        m["fix_rate_recoverable"] = ratio(fix_hr, fix_tr)
        m["unrecoverable_targets"] = unrecoverable
    if entity_spans is not None:
        m["entity_fix_rate"] = ratio(ent_fix_h, ent_fix_t)
        m["entity_keep_rate"] = ratio(ent_keep_h, ent_keep_t)
        m["entity_spans_whisper_wrong"] = ent_fix_t
        m["entity_spans_whisper_right"] = ent_keep_t
    return m
