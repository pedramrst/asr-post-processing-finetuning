"""Word-level edits against the original Whisper transcript: extract them
from any (source, hypothesis) pair, validate them, and apply them back onto
the source deterministically.

This is the core representation for the edit-based correction method (see
src/edit/README.md): a model -- or a rule, or a filter over another model's
full rewrite -- proposes edits; this module is the only thing that ever
builds the final text, so every word outside an accepted edit is the
original Whisper word, byte for byte.

Edits address *word indices* into `source.split()`, not character offsets:
an LLM can't reliably count characters (especially across Persian ZWNJs),
but it can copy a small integer tag it was shown next to each input word.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jiwer


@dataclass
class Edit:
    """Replace source words [start, end) with `replacement`.

    start == end is a pure insertion before word `start`; an empty
    `replacement` is a pure deletion. `original` is the source span's
    words, kept so a proposed edit can be checked against the text it
    claims to change (see validate_edits)."""
    start: int
    end: int
    original: list[str]
    replacement: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        if not self.original:
            return "insert"
        if not self.replacement:
            return "delete"
        return "substitute"


def extract_edits(source: str, hypothesis: str) -> list[Edit]:
    """Minimal edits turning `source` into `hypothesis`, from jiwer's word
    alignment (the same aligner evaluate.py scores with).

    Adjacent non-equal alignment chunks are merged into one edit, so a
    split/merge like "بدنبهشون" -> "بدن بهشون" (a substitute chunk plus an
    insert chunk to jiwer) comes out as the single edit it really is."""
    src_words, hyp_words = source.split(), hypothesis.split()
    if not src_words:
        return [Edit(0, 0, [], hyp_words)] if hyp_words else []
    if not hyp_words:
        return [Edit(0, len(src_words), src_words, [])]
    # jiwer splits on spaces only; str.split() also splits on newlines/tabs
    # (which rewrite models do emit) -- align the space-joined words so both
    # agree on word indices.
    chunks = jiwer.process_words([" ".join(src_words)], [" ".join(hyp_words)]).alignments[0]

    edits: list[Edit] = []
    run = None  # [src_start, src_end, hyp_start, hyp_end] of the current non-equal run
    for chunk in chunks:
        if chunk.type == "equal":
            if run:
                edits.append(_edit_from_run(run, src_words, hyp_words))
                run = None
            continue
        if run is None:
            run = [chunk.ref_start_idx, chunk.ref_end_idx, chunk.hyp_start_idx, chunk.hyp_end_idx]
        else:
            run[1], run[3] = chunk.ref_end_idx, chunk.hyp_end_idx
    if run:
        edits.append(_edit_from_run(run, src_words, hyp_words))
    return edits


def _edit_from_run(run, src_words, hyp_words) -> Edit:
    s0, s1, h0, h1 = run
    return Edit(s0, s1, src_words[s0:s1], hyp_words[h0:h1])


def validate_edits(source_words: list[str], edits: list[Edit]) -> tuple[list[Edit], list[tuple[Edit, str]]]:
    """Splits `edits` into (valid, rejected-with-reason).

    Rejects out-of-range spans, an `original` that doesn't match the
    source words it points at, no-op edits, and any edit overlapping an
    earlier-accepted one (first come, first kept -- callers order edits by
    priority if that matters). Two insertions at the same index, or an
    insertion at a substitution's boundary, don't overlap."""
    valid: list[Edit] = []
    rejected: list[tuple[Edit, str]] = []
    taken: list[tuple[int, int]] = []
    n = len(source_words)
    for e in edits:
        if not (0 <= e.start <= e.end <= n):
            rejected.append((e, "out_of_range"))
        elif source_words[e.start:e.end] != e.original:
            rejected.append((e, "original_mismatch"))
        elif e.original == e.replacement:
            rejected.append((e, "no_op"))
        elif any(e.start < b and a < e.end for a, b in taken) or (
            e.start == e.end and any(a < e.start < b for a, b in taken)
        ):
            rejected.append((e, "overlap"))
        else:
            valid.append(e)
            taken.append((e.start, e.end))
    return valid, rejected


def apply_edits(source: str, edits: list[Edit]) -> str:
    """Applies validated, non-overlapping `edits` to `source` right to left
    (so earlier indices stay valid), returning the new transcript. Raises
    on an invalid edit set -- run validate_edits first for untrusted input."""
    words = source.split()
    valid, rejected = validate_edits(words, edits)
    if rejected:
        raise ValueError(f"apply_edits got invalid edits: {[(r, e) for e, r in rejected]}")
    # Right to left; among same-start edits, apply the one with the larger
    # end first so a pure insertion at i lands before a substitution at i.
    for e in sorted(valid, key=lambda e: (e.start, e.end), reverse=True):
        words[e.start:e.end] = e.replacement
    return " ".join(words)
