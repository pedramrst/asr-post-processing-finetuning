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

import re
from dataclasses import dataclass, field

import jiwer

# Persian number words (cardinals, ordinals, and common colloquial spelling
# variants -- e.g. "شونزده" for "شانزده") and plain digits. An edit touching
# any of these is never applied, by anyone: not training data, not the
# verifier, not a trained model's own output at inference (every one of
# those paths runs through validate_edits below). This is deliberately
# blunt -- it excludes ANY edit touching a number-word token, not just ones
# that actually change its value, because telling "same value, reworded"
# apart from "different value" is exactly the judgment call that shouldn't
# be trusted on sensitive fields (order codes, amounts, tracking numbers).
# See src/edit/README.md's "Never changes numbers".
NUMBER_WORDS = frozenset(
    "صفر یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده پانزده شانزده هفده "
    "هجده نوزده بیست سی چهل پنجاه شصت هفتاد هشتاد نود صد دویست سیصد چهارصد پانصد ششصد "
    "هفتصد هشتصد نهصد هزار میلیون میلیارد "
    "شونزده سینزده دوازه شیش شیشصد نونزده هشصد هیجده هیفده پونزده پونصد چار چارده چارصد یازه "
    "اول دوم سوم چهارم پنجم ششم هفتم هشتم نهم دهم".split()
)
_DIGIT = re.compile(r"[0-9۰-۹]")


def touches_number(words: list[str]) -> bool:
    """True if any word is a digit or a Persian number word (see NUMBER_WORDS)."""
    return any(_DIGIT.search(w) or w.strip("‌") in NUMBER_WORDS for w in words)


def _squash(words: list[str]) -> str:
    return "".join(words).replace("‌", "")


def changes_number(original: list[str], replacement: list[str]) -> bool:
    """touches_number(), but exempts a pure re-segmentation: the exact same
    characters, just split/joined into different word boundaries (e.g.
    rules.py's glue-splitting can turn "خودتونهشصت" into "خودتونه" + "شصت" --
    same characters throughout, so no digit's value could have changed).
    Only this, not touches_number itself, gates validate_edits: a
    resegmentation is never rejected just because a number word happens to
    fall out of it, but any edit that actually changes the characters of a
    number word still is."""
    return (touches_number(original) or touches_number(replacement)) and _squash(original) != _squash(replacement)


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


def validate_edits(source_words: list[str], edits: list[Edit], allow_numbers: bool = False
                   ) -> tuple[list[Edit], list[tuple[Edit, str]]]:
    """Splits `edits` into (valid, rejected-with-reason).

    Rejects out-of-range spans, an `original` that doesn't match the
    source words it points at, no-op edits, an edit that changes a
    number's value (see changes_number -- unless `allow_numbers` is set;
    only rules.py's own deterministic, already-reviewed preprocessing
    passes that, never anything LLM-judged, verifier-decided, or
    model-generated), and any edit overlapping an earlier-accepted one
    (first come, first kept -- callers order edits by priority if that
    matters). Two insertions at the same index, or an insertion at a
    substitution's boundary, don't overlap."""
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
        elif not allow_numbers and changes_number(e.original, e.replacement):
            rejected.append((e, "touches_number"))
        elif any(e.start < b and a < e.end for a, b in taken) or (
            e.start == e.end and any(a < e.start < b for a, b in taken)
        ):
            rejected.append((e, "overlap"))
        else:
            valid.append(e)
            taken.append((e.start, e.end))
    return valid, rejected


def apply_edits(source: str, edits: list[Edit], allow_numbers: bool = False) -> str:
    """Applies validated, non-overlapping `edits` to `source` right to left
    (so earlier indices stay valid), returning the new transcript. Raises
    on an invalid edit set -- run validate_edits first for untrusted input.
    `allow_numbers`: see validate_edits."""
    words = source.split()
    valid, rejected = validate_edits(words, edits, allow_numbers=allow_numbers)
    if rejected:
        raise ValueError(f"apply_edits got invalid edits: {[(r, e) for e, r in rejected]}")
    # Right to left; among same-start edits, apply the one with the larger
    # end first so a pure insertion at i lands before a substitution at i.
    for e in sorted(valid, key=lambda e: (e.start, e.end), reverse=True):
        words[e.start:e.end] = e.replacement
    return " ".join(words)
