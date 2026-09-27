"""Entity grounding for the edit-based method: which names/brands a call
could plausibly mention, and a model-free rule that snaps near-miss Whisper
words onto them.

Two very different sources, kept strictly apart:

  * crm_candidates() -- inference-time evidence. The call's CRM record
    (customer name, case owner name) exists before transcription, so a
    corrector may see it. This is the only grounding the step-1 filters and
    crm_snap_edits() use.
  * reference_entity_spans() -- evaluation labels only. CRM names plus the
    Gemini entity extractions in data/output-backup/, which were run over
    the *reference* transcripts; feeding them to a corrector would leak the
    answer. Used solely to score entity recovery (scoring.py).

No product/brand catalog exists yet, so product and brand names are only
evaluated, not grounded -- see README, step 1 finding 4.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from build_entity_eval_slice import persian_name_words  # noqa: E402
from data import _phonetic_similarity  # noqa: E402
from edits import Edit  # noqa: E402

MIN_NAME_WORD_LEN = 3  # same floor the entity eval slice uses

# Gemini (group, category, subtype) triples that are real named entities
# worth protecting/recovering -- names, brands, shops, products, places.
# Times, amounts, order/phone numbers, roles, and internal jargon are left
# out: they're either not entities or scored better by plain WER.
GEMINI_ENTITY_TYPES = {
    ("people_org", "personal_info", "family"),
    ("people_org", "personal_info", "name"),
    ("people_org", "business", "brand"),
    ("people_org", "business", "shop_name"),
    ("commerce", "business", "brand"),
    ("commerce", "product", "name"),
    ("commerce", "product", "model"),
    ("commerce", "product", "service"),
    ("commerce", "product", "plan"),
    ("context", "place", "city"),
    ("context", "place", "neighborhood"),
}


def crm_candidates(crm_metadata: str | dict | None) -> set[str]:
    """Persian name words from a call's CRM record (customer + case owner)."""
    if not crm_metadata:
        return set()
    crm = json.loads(crm_metadata) if isinstance(crm_metadata, str) else crm_metadata
    return persian_name_words(crm, MIN_NAME_WORD_LEN)


def _squash(text: str) -> str:
    """Space/ZWNJ-free form, so "نمونه پور" and "نمونه‌پور" compare equal."""
    return text.replace(" ", "").replace("‌", "")


def span_similarity(a_words: list[str], b_words: list[str]) -> float:
    """data._phonetic_similarity over the two spans with spacing removed --
    a split/merge ("بدنبهشون" vs "بدن بهشون") scores 1.0, a homophone
    substitution ("زفری" vs "ظفری") scores 1.0, an unrelated word low."""
    return _phonetic_similarity(_squash(" ".join(a_words)), _squash(" ".join(b_words)))


def crm_snap_edits(
    source: str,
    candidates: set[str],
    corpus_freq: Counter,
    min_similarity: float = 0.75,
    max_source_freq: int = 20,
    max_pair_word_freq: int = 500,
) -> list[Edit]:
    """Model-free entity correction: replace a Whisper word (or adjacent
    word pair, for split names) with a CRM name word it's a phonetic near
    miss of.

    Gated so it can't turn an ordinary word into a name: the Whisper side
    must be rare in the training corpus (<= max_source_freq occurrences;
    a real misrecognition of a name is usually a rare or non-word, while a
    common word that happens to look like a name is usually just that
    word), and must not already be the name. A pair is only joined when at
    least one of its words is not very common (<= max_pair_word_freq) --
    otherwise "به من" ("to me") becomes the name "بهمن"."""
    words = source.split()
    if not candidates or not words:
        return []
    squashed_candidates = {_squash(c): c for c in candidates}
    edits: list[Edit] = []
    i = 0
    while i < len(words):
        # A name Whisper split into two words ("نمونه پور") -> join it.
        pair = words[i:i + 2]
        pair_match = squashed_candidates.get(_squash(" ".join(pair))) if len(pair) == 2 else None
        if pair_match and not any(w in candidates for w in pair) and min(
            corpus_freq.get(w, 0) for w in pair
        ) <= max_pair_word_freq:
            edits.append(Edit(i, i + 2, pair, [pair_match], meta={"source": "crm_snap", "similarity": 1.0}))
            i += 2
            continue
        # A rare single word that's a phonetic near miss of a name.
        w = words[i]
        if w not in candidates and len(w) >= MIN_NAME_WORD_LEN and corpus_freq.get(w, 0) <= max_source_freq:
            scored = [(span_similarity([w], [c]), c) for c in candidates if _squash(c) != _squash(w)]
            sim, cand = max(scored, default=(0.0, None))
            if cand and sim >= min_similarity:
                edits.append(Edit(i, i + 1, [w], [cand], meta={"source": "crm_snap", "similarity": round(sim, 3)}))
        i += 1
    return edits


def load_gemini_entities(entities_dir: str | Path) -> dict[str, list[str]]:
    """call_id -> entity span strings of the types in GEMINI_ENTITY_TYPES."""
    out: dict[str, list[str]] = {}
    for path in Path(entities_dir).glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        spans = []
        for group in (d.get("groups") or {}).values():
            for e in group.get("entities") or []:
                key = (group.get("group_name"), e.get("category"), e.get("subtype"))
                span = (e.get("span") or "").strip()
                if key in GEMINI_ENTITY_TYPES and len(span) >= MIN_NAME_WORD_LEN:
                    spans.append(span)
        out[d.get("call_id") or path.stem] = spans
    return out


def reference_entity_spans(reference: str, crm_words: set[str], gemini_spans: list[str]) -> list[str]:
    """Evaluation-only entity labels for one row: CRM name words and Gemini
    spans that actually occur (whole-word, spacing-insensitive) in the
    reference, deduplicated. See the module docstring on why these never
    reach a corrector."""
    from scoring import _contains_span  # local import: scoring imports heavy eval deps

    seen, spans = set(), []
    for span in sorted(crm_words) + gemini_spans:
        key = _squash(span)
        if key not in seen and _contains_span(reference, span):
            seen.add(key)
            spans.append(span)
    return spans
