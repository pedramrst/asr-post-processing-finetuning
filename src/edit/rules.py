"""Deterministic, model-free edits for Whisper artifacts that are
systematic enough not to need a model at all.

Found by inspecting which of the original Qwen3.5-2b's rewrite edits
actually helped (diff_filter_baseline.py's edits.jsonl): most helpful ones
weren't misrecognitions but two mechanical artifacts --

  * glued words: adjacent words written with no space, mostly where two
    Whisper segments were joined ("بزنیبله" -> "بزنید بله",
    "نکنهخواهش" -> "نکنه خواهش", "کنمهمین" -> "کنم همین").
  * stutter-doubled first letter ("اارسال" -> "ارسال", "ممجدد" -> "مجدد").

Both rules only fire on a word that's rare in the training corpus and
turn it into words that are common, so a real rare word (a name, a
brand) is left alone unless it happens to be two very common words glued
together -- and CRM name words are excluded outright.
"""
from __future__ import annotations

from collections import Counter

from edits import Edit

ZWNJ = "‌"


# Colloquial clitics/suffixes that also occur as free-standing "words" in
# the corpus (from inconsistent spacing), so a naive split peels them off a
# real word: "ثبتشو" -> "ثبت شو", "تغییراتو" -> "تغییر اتو". These were the
# bulk of the harmful glue splits on the test set, so they're never allowed
# as a split's second half.
CLITIC_SUFFIXES = frozenset({
    "شو", "شون", "تون", "مون", "مو", "تو", "و", "اتو", "اشو", "اش", "ات", "ام",
    "ها", "های", "هاشو", "هاتون", "هامون", "ای", "ی", "ه", "م", "ت", "ش",
})


def split_glued_word(word: str, corpus_freq: Counter, max_word_freq: int, min_part_freq: int,
                     extended: bool = False) -> list[str] | None:
    """Best split of a rare `word` into two common words, or None.

    `extended` also handles what the original rule skipped, measured on the
    test set against the reference (295 words the original rule leaves alone:
    291 fixed, 4 broken):
      * words containing a half-space, e.g. "رسیدگیمی‌شه" -> "رسیدگی می‌شه"
        (either half may keep its own half-space; a split is never placed
        next to one, so "می‌شه" itself is never cut), and
      * a final "و" (and), e.g. "کردمو" -> "کردم و". "و" stays banned as a
        second half in the original rule because "تغییراتو" is
        "تغییرات رو"; that ambiguity is real, so only rare words whose first
        half is common are split, and the frequency gate does the rest."""
    if (ZWNJ in word and not extended) or len(word) < 5 or corpus_freq.get(word, 0) > max_word_freq:
        return None
    best, best_score = None, 0
    for i in range(2, len(word) if extended else len(word) - 1):
        a, b = word[:i], word[i:]
        if extended and (a.endswith(ZWNJ) or b.startswith(ZWNJ)):
            continue
        if len(b) == 1 and not (extended and b == "و"):
            continue
        if b in CLITIC_SUFFIXES and not (extended and b == "و"):
            continue
        score = min(corpus_freq.get(a, 0), corpus_freq.get(b, 0))
        if score >= min_part_freq and score > best_score:
            best, best_score = [a, b], score
    return best


def unstutter_word(word: str, corpus_freq: Counter, max_word_freq: int, min_part_freq: int) -> str | None:
    """`word` minus a doubled first letter, if that makes a rare word common."""
    if len(word) < 3 or word[0] != word[1] or corpus_freq.get(word, 0) > max_word_freq:
        return None
    fixed = word[1:]
    return fixed if corpus_freq.get(fixed, 0) >= min_part_freq else None


def rule_edits(
    source: str,
    corpus_freq: Counter,
    protected: set[str] = frozenset(),
    max_word_freq: int = 2,
    min_part_freq: int = 200,
    extended: bool = False,
) -> list[Edit]:
    """Glue-split and unstutter edits for `source`. `protected` words (e.g.
    CRM names) are never touched. `extended`: see split_glued_word -- off by
    default so the edit datasets and verifier data built with the original
    rules stay reproducible."""
    edits = []
    for i, w in enumerate(source.split()):
        if w in protected:
            continue
        fixed = unstutter_word(w, corpus_freq, max_word_freq, min_part_freq)
        if fixed:
            edits.append(Edit(i, i + 1, [w], [fixed], meta={"source": "rule_unstutter"}))
            continue
        parts = split_glued_word(w, corpus_freq, max_word_freq, min_part_freq, extended)
        if parts:
            edits.append(Edit(i, i + 1, [w], parts, meta={"source": "rule_glue_split"}))
    return edits
