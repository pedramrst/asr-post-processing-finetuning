"""Normalizes away Persian dialectal/spacing variation that isn't a real
correction difference, for evaluate.py's *_lenient metrics (fix_rate/
preservation_rate/targeted_score computed after this normalization, instead
of on raw text).

Four mechanisms, each scoped down after testing directly against real data
and this project's confirmed entity words (see build_entity_eval_slice.py)
-- a name/brand getting silently "corrected" into something else is the one
failure mode worth being paranoid about here, since it's exactly the content
the entity slice exists to protect:

  1. hazm.Normalizer() -- ZWNJ/spacing normalization (e.g. "می کنم"/
     "میکنم"/"می‌کنم" all collapse to the same form). Verified clean against
     1,414 confirmed entity spans; one exception found ("میناوند", a
     surname, gets a ZWNJ inserted after "می" as if it were the verb prefix
     -- low-severity since it's an invisible character, not a word swap, and
     self-cancels as long as it's applied identically to both sides being
     compared, which it always is here).

  2. hazm.InformalNormalizer's direct dictionary (iword_map) ONLY --
     deliberately NOT its fallback suffix-guessing heuristic (tries
     stripping endings like ها/و/ا/ه and checking if what's left is a known
     word). That fallback is where real corruption happens: tested directly
     against 1,488 confirmed entity tokens, 42 got "normalized" by it,
     including "آنتونیا" (a name) -> "آنتونی‌ها", "ایکیا" (IKEA) ->
     "ایکی‌ها", "اسپرسو" -> "اسپرس را"/"اسپرس و", "نسرین" (a name) -> "نسرید"
     (not even a real word) -- none of which are direct dictionary entries.
     Restricting to direct hits only gave 9 matches among those same 1,488
     tokens, all correct, e.g. یه->یک, خب->خوب, دیگه->دیگر, تومن->تومان.
     _IWORD_SKIP further excludes "شبا", which hazm's dictionary maps to
     "شب‌ها" ("the nights") but which this call-center domain overwhelmingly
     uses to mean Sheba/IBAN (a bank account number format) instead.

  3. Informal/formal verb-ending equivalence (کنین~کنید, ببینین~ببینید,
     ...) -- our own rule, not hazm's verb-conjugation tables (which
     produced "می‌خواستم" -> "می‌خواهستم", not a real word, when tested on a
     real transcript). Only ever applied to a pair of words an alignment has
     ALREADY decided are a substitution for each other (see
     evaluate.py:_equal_or_equivalent_positions), so unlike a blind
     dictionary/regex rewrite it can't misfire on an unrelated word -- it
     simply never fires unless the *other* side already has a matching
     candidate at the same position. Validated against a 20k-row sample:
     239 distinct pairs, manually inspected, zero false positives after
     requiring a minimum stem length (a naive version matches "این"/"اید",
     stem "ا", one character).

  4. را/رو equivalence -- same already-aligned-pair mechanism and reasoning
     as (3), for the bare object marker written formally vs. colloquially
     ("را" vs. "رو") and for a word with either one attached directly
     ("چیزو" ~ "چیز" + رو). Validated the same way: 691 bare + 1,519
     attached-word pairs in the same sample, inspected, zero false
     positives.

(3) and (4) are NOT applied by normalize_lenient() below -- they only make
sense as an equivalence check between two specific already-aligned words,
not as a text rewrite (a blind "word ending in و means را/رو was attached"
rule would corrupt real word-final-و words like تو/او/دو/نو). See
words_equivalent() and evaluate.py's use of it.
"""
from __future__ import annotations

import hazm

_normalizer = hazm.Normalizer()
_informal = hazm.InformalNormalizer()

# See module docstring (2) -- a real domain term this dictionary entry
# collides with.
_IWORD_SKIP = frozenset({"شبا"})

# See module docstring (3) -- avoids matching on a near-empty stem (e.g.
# "این"/"اید", stem "ا").
_VERB_ENDING_MIN_STEM_LEN = 2


def normalize_spacing(text: str) -> str:
    """ZWNJ/می-spacing normalization only -- module docstring (1)."""
    return _normalizer.normalize(text)


def normalize_lenient(text: str) -> str:
    """Full text-level lenient normalization: (1) then (2). Word-count
    preserving isn't guaranteed (hazm.Normalizer can join "می کنم" into one
    token), which is fine -- callers realign the normalized text fresh
    rather than reusing indices from a raw-text alignment."""
    text = normalize_spacing(text)
    return " ".join(
        _informal.iword_map.get(w, w) if w not in _IWORD_SKIP else w for w in text.split()
    )


def _verb_ending_equivalent(a: str, b: str) -> bool:
    """Module docstring (3): same stem, informal -ین vs. formal -ید ending."""
    min_len = _VERB_ENDING_MIN_STEM_LEN + 2
    if len(a) < min_len or len(b) < min_len:
        return False
    if a.endswith("ید") and b.endswith("ین"):
        return a[:-2] == b[:-2]
    if a.endswith("ین") and b.endswith("ید"):
        return a[:-2] == b[:-2]
    return False


def _raa_attached_equivalent(a: str, b: str) -> bool:
    """Module docstring (4): either the bare object marker written the
    formal way vs. the colloquial way ("را" vs. "رو"), or the same word
    with either one attached directly ("چیزو" ~ "چیز" + رو). Validated
    directly against a 20k-row sample: 691 bare را/رو substitutions, 1,519
    attached-word ones, both inspected, no false positives."""
    if {a, b} == {"را", "رو"}:
        return True
    a_z, b_z = a.rstrip("‌"), b.rstrip("‌")
    return (
        a_z + "و" == b_z or b_z + "و" == a_z
        or a_z + "رو" == b_z or b_z + "رو" == a_z
    )


def words_equivalent(a: str, b: str) -> bool:
    """True if two words that an alignment has already matched as a
    substitution pair should be scored as equal instead -- combines (3) and
    (4). Not meant to be called on arbitrary unaligned word pairs (see
    module docstring's closing note)."""
    return a == b or _verb_ending_equivalent(a, b) or _raa_attached_equivalent(a, b)
