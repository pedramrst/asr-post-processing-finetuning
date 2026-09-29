"""Text-level helpers shared by the edit-method scripts: punctuation
stripping, "is this edit only a spelling/dialect variant" detection, and a
clitic-joining normalization for scoring."""
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


# Colloquial clitics written attached to the previous word or as their own
# word: "همینو" ~ "همین رو" (object marker), "پیگیریشو" ~ "پیگیریش رو",
# "الانم" ~ "الان هم" ("also"). The verifier audit found these made up most
# changes the reference counted as fixes but the judges called "not an error"
# -- Soniox writes the clitic apart, Whisper attached. As separate words they
# join onto the previous word here; the attached form is what's left alone.
_JOINABLE = {"رو": "و", "را": "و", "هم": "م"}


def join_clitics(words: list[str]) -> list[str]:
    """Separate رو/را/هم attached to the previous word ("همین رو" -> "همینو",
    "الان هم" -> "الانم"), so both spellings of a clitic compare equal. Only
    meant for comparing two texts that both went through it."""
    out: list[str] = []
    for w in words:
        if out and w in _JOINABLE and not out[-1] in _JOINABLE:
            out[-1] = out[-1].rstrip("‌") + _JOINABLE[w]
        else:
            out.append(w)
    return out


# Arabic-script letter variants hazm's normalizer doesn't fold (it does
# ك->ک and ي->ی; these four don't survive it -- checked directly). None of
# these ever change a word's meaning, only which Arabic-inherited letter
# spells the same sound: "قوة"/"قوه", "مؤمن" spelling, "رئیس"/"رییس",
# "أحمد"/"احمد". Symmetric, applied to both sides alike.
_LETTER_FOLD = str.maketrans({"ة": "ه", "ؤ": "و", "ئ": "ی", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})

# Colloquial spellings of a number that mean the exact same value -- never a
# different number, only a different way of saying the same one (e.g.
# "شونزده" and "شانزده" both mean 16). Deliberately small and only ever
# checked word-for-word against another number word already in the same
# position, so it can't turn one value into another the way a stray digit
# edit could.
_NUMBER_VARIANT = {
    "دوازه": "دوازده", "سینزده": "سیزده", "شونزده": "شانزده", "شیش": "شش", "شیشصد": "ششصد",
    "نونزده": "نوزده", "هشصد": "هشتصد", "هیجده": "هجده", "هیفده": "هفده", "پونزده": "پانزده",
    "پونصد": "پانصد", "چار": "چهار", "چارده": "چهارده", "چارصد": "چهارصد", "یازه": "یازده",
}


def fold_letters(word: str) -> str:
    return word.translate(_LETTER_FOLD)


def fold_number_variant(word: str) -> str:
    return _NUMBER_VARIANT.get(word, word)


def _canonical_word(word: str) -> str:
    return fold_number_variant(fold_letters(word))


def normalize_clitics(text: str) -> str:
    """Comparison-only canonicalization for scoring's *_clitic metrics:
    letter-variant folding + number-spelling folding, word by word, then
    join_clitics. Never applied to real text, never touches a number's
    value (only its spelling) -- see fold_number_variant."""
    return " ".join(join_clitics([_canonical_word(w) for w in text.split()]))


def is_style_edit(e: Edit) -> bool:
    """Only a spelling/spacing/dialect variant, not a different word:
    ZWNJ/spacing-only, equal after persian_normalize.normalize_lenient,
    word-by-word persian_normalize.words_equivalent (verb endings, را/رو),
    word-by-word letter-variant/number-spelling folding (fold_letters/
    fold_number_variant), or the same once separate clitics are joined
    (join_clitics: "همینو" ~ "همین رو")."""
    a, b = " ".join(e.original), " ".join(e.replacement)
    if squash(a) == squash(b) or normalize_lenient(a) == normalize_lenient(b):
        return True
    if len(e.original) == len(e.replacement) and all(
            words_equivalent(x, y) or _canonical_word(x) == _canonical_word(y)
            for x, y in zip(e.original, e.replacement)):
        return True
    ja, jb = join_clitics(e.original), join_clitics(e.replacement)
    return len(ja) == len(jb) and all(squash(x) == squash(y) or words_equivalent(x, y) for x, y in zip(ja, jb))
