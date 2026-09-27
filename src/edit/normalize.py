"""Text-level helpers shared by the edit-method scripts: punctuation
stripping and "is this edit only a spelling/dialect variant" detection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from edits import Edit  # noqa: E402
from persian_normalize import normalize_lenient, words_equivalent  # noqa: E402

# Neither Whisper's output nor the `text` reference is punctuated, but a
# rewrite model or the raw Soniox text often is ("کنم" -> "کنم.") -- every
# such edit reads as a broken word yet sounds identical, so it would pass a
# phonetic filter. "-" and "/" are kept since they occur inside model
# numbers/codes.
_PUNCT = str.maketrans("", "", ".,،؛;:!?؟«»\"“”()[]…")


def strip_punctuation(text: str) -> str:
    return " ".join(w for w in text.translate(_PUNCT).split() if w)


def squash(text: str) -> str:
    """Space/ZWNJ-free form, so "نمونه پور" and "نمونه‌پور" compare equal."""
    return text.replace(" ", "").replace("‌", "")


def is_style_edit(e: Edit) -> bool:
    """Only a spelling/spacing/dialect variant, not a different word:
    ZWNJ/spacing-only, equal after persian_normalize.normalize_lenient, or
    word-by-word persian_normalize.words_equivalent (verb endings, را/رو)."""
    a, b = " ".join(e.original), " ".join(e.replacement)
    if squash(a) == squash(b) or normalize_lenient(a) == normalize_lenient(b):
        return True
    return len(e.original) == len(e.replacement) and all(
        words_equivalent(x, y) for x, y in zip(e.original, e.replacement))
