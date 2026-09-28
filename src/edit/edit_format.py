"""The edit model's text format: how a Whisper window is shown to the model
and how its corrections come back.

Input (user turn): the window's words, each tagged with its 1-based number,
plus the call's known CRM names when there are any:

    Known names: نمونه‌پور، مریم
    [1]سلام [2]خانم [3]نمونپور [4]وقت [5]بخیر ...

Output (assistant turn): one line per correction, in word order, or NONE:

    3 نمونپور → نمونه‌پور
    12-13 به سطح → بسته‌ت رو

"12-13" is an inclusive range of word numbers. The original words after the
numbers are optional (EditConfig.target_includes_original, on by default):
they cost a few tokens but let parse_edits() reject a line whose numbers
point at different words than the model thinks it's changing, instead of
silently corrupting the transcript. Scope is v1: substitutions only (a
replacement may be a different number of words, e.g. a split or merged
word), never pure insertions or deletions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from edits import Edit, validate_edits

SYSTEM_PROMPT = (
    "You correct speech-recognition errors in transcripts of Persian customer-service phone calls "
    "(Digikala, an online store). You get a stretch of the transcript with every word numbered, and "
    "sometimes the names of the people on the call from the store's records.\n"
    "Find only clear recognition errors: a word that doesn't fit the sentence because the recognizer "
    "misheard something that sounds similar, or a misheard name, brand or product. Leave everything else "
    "exactly as it is -- do not change spelling, half-spaces, or colloquial forms (میشه، می‌خوام، رو are "
    "correct), do not add or remove words, and do not rewrite the sentence.\n"
    "Output one line per correction, in order: the word number or range (e.g. 12-13), the original words, "
    "→, and the corrected words. If nothing needs correcting, output only NONE."
)
SYSTEM_PROMPT_NO_ORIGINAL = SYSTEM_PROMPT.replace(
    "the word number or range (e.g. 12-13), the original words, →, and the corrected words",
    "the word number or range (e.g. 12-13), →, and the corrected words",
)
NONE = "NONE"
ARROW = "→"

# "12-13 original words → replacement" or "12 → replacement"
_LINE = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s*(.*?)\s*" + ARROW + r"\s*(.*?)\s*$")


def system_prompt(include_original: bool = True) -> str:
    return SYSTEM_PROMPT if include_original else SYSTEM_PROMPT_NO_ORIGINAL


def render_input(words: list[str], crm_names: list[str] | None = None) -> str:
    numbered = " ".join(f"[{i}]{w}" for i, w in enumerate(words, start=1))
    if crm_names:
        return f"Known names: {'، '.join(crm_names)}\n{numbered}"
    return numbered


def render_target(edits: list[dict] | list[Edit], include_original: bool = True) -> str:
    """Edits (0-based [start, end) word spans, as in train.jsonl) -> output text."""
    lines = []
    for e in sorted(edits, key=lambda e: _get(e, "start")):
        start, end = _get(e, "start"), _get(e, "end")
        span = f"{start + 1}" if end - start == 1 else f"{start + 1}-{end}"
        original = _words(_get(e, "original"))
        replacement = _words(_get(e, "replacement"))
        lines.append(f"{span} {original} {ARROW} {replacement}" if include_original else f"{span} {ARROW} {replacement}")
    return "\n".join(lines) if lines else NONE


@dataclass
class ParseResult:
    edits: list[Edit]
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (line, reason)
    said_none: bool = False


def parse_edits(text: str, words: list[str]) -> ParseResult:
    """Model output -> validated edits against `words`. Lines that don't
    parse, point outside the window, name different original words than
    the ones at those positions, are no-ops, or overlap an earlier line are
    rejected (and reported), never applied."""
    text = (text or "").strip()
    if not text or text.upper() == NONE:
        return ParseResult([], said_none=True)
    candidates, rejected = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper() == NONE:
            continue
        m = _LINE.match(line)
        if not m:
            rejected.append((line, "unparseable"))
            continue
        first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        start, end = first - 1, last
        if not (0 <= start < end <= len(words)):
            rejected.append((line, "out_of_range"))
            continue
        claimed, replacement = m.group(3).split(), m.group(4).split()
        if claimed and claimed != words[start:end]:
            rejected.append((line, "original_mismatch"))
            continue
        if not replacement:
            rejected.append((line, "empty_replacement"))  # deletions are out of scope for v1
            continue
        candidates.append(Edit(start, end, words[start:end], replacement))
    valid, invalid = validate_edits(words, candidates)
    rejected += [(f"{e.start + 1}-{e.end} {' '.join(e.replacement)}", reason) for e, reason in invalid]
    return ParseResult(valid, rejected)


def _get(e, key):
    return e[key] if isinstance(e, dict) else getattr(e, key)


def _words(v) -> str:
    return v if isinstance(v, str) else " ".join(v)
