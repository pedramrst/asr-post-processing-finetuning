#!/usr/bin/env python3
"""Step 2a of the edit-based method: turn training rows into individual
Whisper-vs-Soniox *differences* that an LLM judge (llm_judge.py) or a
person (make_label_sheet.py) can rule on one at a time.

Per unique training row (the prebuilt dataset repeats upsampled rows; they
are deduplicated here):

  1. Apply rules.py's deterministic fixes (glued words, doubled letters) to
     the Whisper text first -- those need no judge.
  2. Align the result against Soniox's `text` (edits.extract_edits).
  3. Split spans that mix a style-only and a real difference (split_mixed).
  4. Mark style-only differences (normalize.is_style_edit -- ZWNJ/spacing,
     informal/formal endings, را/رو) as `style`, filler-only ones as
     `filler`, and ones crowded by other differences as `garbled`. None of
     these are judged: they're never correction targets under the "fix real
     errors" objective, or can't be decided from text.
  5. Everything else becomes a difference item with features and a
     `stratum` used for sampling:

       entity          Soniox side has a CRM name, or Whisper side is a
                       near miss of one, or Soniox side is a rare word
       phonetic_sub    substitution, both sides sound alike (>= 0.6)
       other_sub       substitution, sides don't sound alike
       whisper_dropped Soniox has words Whisper has nothing for
       whisper_extra   Whisper has words Soniox has nothing for
       long            a side longer than MAX_ITEM_WORDS (likely
                       misalignment or a skipped stretch of audio)
       style, filler, garbled, number   see 4 (not judged)

Writes <output-dir>/rows.jsonl (one per unique row: Whisper after rules,
Soniox text, CRM names, diff counts) and <output-dir>/diffs.jsonl (one per
difference). Row and diff ids are stable content hashes, so reruns and
judge caches line up.

Example:
  python3 src/edit/build_diffs.py
  python3 src/edit/build_diffs.py --max-rows 5000   # quick look
  python3 src/edit/build_diffs.py --max-rows 8000 --output-dir outputs/edit/targets_v2 \
      --exclude-calls-from outputs/edit/targets/rows.jsonl   # a fresh pool of new calls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from tqdm import tqdm  # noqa: E402

from corpus import CACHE_DIR, load_corpus_freq  # noqa: E402
from edits import Edit, apply_edits, changes_number, extract_edits  # noqa: E402
from evaluate import _load_hub_columns_pruned  # noqa: E402
from grounding import MIN_NAME_WORD_LEN, crm_candidates, span_similarity  # noqa: E402
from normalize import is_style_edit, squash  # noqa: E402
from rules import rule_edits  # noqa: E402

COLUMNS = ["call_id", "channel", "assembled", "text_whisper", "text", "crm_context", "wer_whisper"]
CONTEXT_WORDS = 12
MAX_ITEM_WORDS = 4
PHONETIC_SUB_MIN_SIMILARITY = 0.6
RARE_WORD_MAX_FREQ = 3
CRM_NEAR_MIN_SIMILARITY = 0.75
# A difference with this many other (non-style, non-filler) differences
# within GARBLED_WINDOW words sits in a stretch Whisper got badly wrong: its
# context is unreadable, so neither a person nor a text-only judge can rule
# on it (round 1 of hand labelling). Never judged; stays Whisper's text.
NOT_JUDGED = ("style", "filler", "garbled", "number")
GARBLED_WINDOW = 3
GARBLED_MIN_NEIGHBOURS = 2
# Filler/discourse words. A difference made only of these (a dropped "بله",
# "و" vs nothing, "آه" vs "الو") can't be decided from text and doesn't
# change the meaning: round 1's hand labels put 17 of 22 such items as
# "doesn't matter"/"unsure". Never judged.
FILLER_WORDS = frozenset({
    "بله", "آره", "آها", "اها", "الو", "آه", "اه", "ام", "امم", "اممم", "خب", "خوب", "و", "رو", "را",
    "هم", "دیگه", "یعنی", "حالا", "مرسی", "ممنون", "باشه", "چشم", "نه", "آقا", "خانم", "اینکه", "که",
    "این", "اون", "یه", "یک", "ببخشید", "بفرمایید", "بفرمایین", "عرض", "حتما", "الان", "جان",
    # Acknowledgement/interjection words beyond "بله" -- found as ordinary
    # single-word insertions/deletions (one side transcribed the
    # interjection, the other didn't), not caught by the words above.
    "اوکی", "اکی", "آهان", "اهان", "اهوم", "اوهوم", "بلی", "بعله",
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-repo", default="PedramR/ASR_Post-processing-dataset")
    p.add_argument("--output-dir", default="outputs/edit/targets")
    p.add_argument("--max-rows", type=int, default=None, help="Random sample of unique rows (default: all).")
    p.add_argument("--exclude-calls-from", default=None,
                   help="Comma-separated rows.jsonl files from earlier builds: their calls are left out, so a new "
                        "pool (e.g. for more training data) never overlaps windows already made from them.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def stable_id(*parts) -> str:
    return hashlib.sha1("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def load_unique_rows(train_repo: str) -> list[dict]:
    """Unique (call_id, channel, text_whisper) rows, cached locally since the
    column-pruned Hub read takes several minutes."""
    cache = CACHE_DIR / "train_unique_rows.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.open(encoding="utf-8")]
    print(f"Loading {COLUMNS} from {train_repo} (cached to {cache})...")
    table = _load_hub_columns_pruned(train_repo, COLUMNS, "train").data.table.to_pylist()
    seen, rows = set(), []
    for r in table:
        key = (r["call_id"], r["channel"], r["text_whisper"])
        if r["text_whisper"] and r["text"] and key not in seen:
            seen.add(key)
            rows.append(r)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")  # written whole, then renamed: a crash never leaves a partial cache
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            # crm_context carries datetimes (e.g. case created_on) -- stored as strings.
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(cache)
    return [json.loads(line) for line in cache.open(encoding="utf-8")]


def split_mixed(e: Edit, max_words: int = 8) -> list[Edit]:
    """Splits an aligned span that mixes a style-only difference with a real
    one -- e.g. Whisper "نمونه‌پور بله یه" vs Soniox "نمونه پور" is a spacing
    variant (نمونه‌پور ~ نمونه پور) plus two extra words -- by peeling off a
    style-equivalent prefix or suffix. Round 1 of hand labelling showed such
    spans can't be given a single verdict. Spans whose two sides match as a
    whole (e.g. a split/merged word) are left as one edit."""
    o, r = e.original, e.replacement
    if len(o) + len(r) <= 2 or len(o) > max_words or len(r) > max_words:
        return [e]
    for i in range(1, len(o) + 1):
        for j in range(1, len(r) + 1):
            if (i, j) == (len(o), len(r)):
                continue
            if is_style_edit(Edit(0, i, o[:i], r[:j])):  # style prefix
                return [Edit(e.start, e.start + i, o[:i], r[:j])] + split_mixed(
                    Edit(e.start + i, e.end, o[i:], r[j:]), max_words)
            if is_style_edit(Edit(0, i, o[-i:], r[-j:])):  # style suffix
                return split_mixed(Edit(e.start, e.end - i, o[:-i], r[:-j]), max_words) + [
                    Edit(e.end - i, e.end, o[-i:], r[-j:])]
    return [e]


def is_filler_edit(e: Edit) -> bool:
    """Every word on both sides is a filler/discourse word (FILLER_WORDS)."""
    words = e.original + e.replacement
    return bool(words) and all(w in FILLER_WORDS for w in words)


def stratum(kind: str, whisper_span: list[str], soniox_span: list[str], sim: float,
            crm_hit: bool, crm_near: bool, soniox_min_freq: int | None, neighbours: int) -> str:
    if max(len(whisper_span), len(soniox_span)) > MAX_ITEM_WORDS:
        return "long"
    if neighbours >= GARBLED_MIN_NEIGHBOURS:
        return "garbled"
    if crm_hit or crm_near or (kind == "substitute" and soniox_min_freq is not None
                               and soniox_min_freq <= RARE_WORD_MAX_FREQ):
        return "entity"
    if kind == "insert":
        return "whisper_dropped"
    if kind == "delete":
        return "whisper_extra"
    return "phonetic_sub" if sim >= PHONETIC_SUB_MIN_SIMILARITY else "other_sub"


_FREQ = None  # set per worker by _init


def _init(freq):
    global _FREQ
    _FREQ = freq


def process_row(row: dict) -> tuple[dict, list[dict]]:
    freq = _FREQ
    names = crm_candidates(row.get("crm_context"))
    row_id = stable_id(row["call_id"], row["channel"], row["text_whisper"])
    rules = rule_edits(row["text_whisper"], freq, protected=names)
    # rules.py's own fixes are deterministic and already reviewed (see its
    # docstring), not LLM-judged or model-generated -- allowed to touch a
    # number (e.g. unstuttering "ببیست" -> "بیست") since they only ever
    # remove a doubled letter or split a glued word, never change a digit.
    whisper = apply_edits(row["text_whisper"], rules, allow_numbers=True)
    w_words = whisper.split()
    soniox = " ".join(row["text"].split())

    # Context comes from the Whisper side only: outside an edit both
    # transcripts agree (up to other nearby diffs), so "left + span + right"
    # shows either version.
    edits = [piece for e in extract_edits(whisper, soniox) for piece in split_mixed(e)]
    style = [is_style_edit(e) for e in edits]
    filler = [not st and is_filler_edit(e) for e, st in zip(edits, style)]
    numeric = [not st and not fl and changes_number(e.original, e.replacement)
               for e, st, fl in zip(edits, style, filler)]
    content = [(e.start, e.end) for e, st, fl, nm in zip(edits, style, filler, numeric) if not st and not fl and not nm]

    diffs = []
    counts = Counter()
    for e, st, fl, nm in zip(edits, style, filler, numeric):
        sim = span_similarity(e.original, e.replacement) if e.original and e.replacement else 0.0
        crm_hit = any(w in names for w in e.replacement)
        crm_near = bool(e.original) and any(
            span_similarity(e.original, [n]) >= CRM_NEAR_MIN_SIMILARITY and squash(" ".join(e.original)) != squash(n)
            for n in names if len(n) >= MIN_NAME_WORD_LEN)
        s_freqs = [freq.get(w, 0) for w in e.replacement]
        neighbours = sum(1 for a, b in content if (a, b) != (e.start, e.end)
                         and a < e.end + GARBLED_WINDOW and b > e.start - GARBLED_WINDOW)
        if st:
            strat = "style"
        elif fl:
            strat = "filler"
        elif nm:
            strat = "number"
        else:
            strat = stratum(e.kind, e.original, e.replacement, sim, crm_hit, crm_near,
                            min(s_freqs) if s_freqs else None, neighbours)
        counts[strat] += 1
        diffs.append({
            "id": stable_id(row_id, e.start, e.end, " ".join(e.replacement)),
            "row_id": row_id,
            "call_id": row["call_id"],
            "start": e.start, "end": e.end,
            "whisper_span": " ".join(e.original),
            "soniox_span": " ".join(e.replacement),
            "kind": e.kind, "stratum": strat,
            "similarity": round(sim, 3),
            "whisper_min_freq": min((freq.get(w, 0) for w in e.original), default=None),
            "soniox_min_freq": min(s_freqs) if s_freqs else None,
            "crm_hit": crm_hit, "crm_near": crm_near, "crm_names": sorted(names),
            "neighbours": neighbours,
            "left": " ".join(w_words[max(0, e.start - CONTEXT_WORDS):e.start]),
            "right": " ".join(w_words[e.end:e.end + CONTEXT_WORDS]),
        })
    row_out = {
        "row_id": row_id, "call_id": row["call_id"], "channel": row["channel"],
        "assembled": row["assembled"], "whisper_raw": row["text_whisper"], "whisper": whisper,
        "soniox": soniox, "crm_names": sorted(names), "rule_edits": len(rules),
        "n_words": len(w_words), "diff_counts": dict(counts),
    }
    return row_out, diffs


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Full-corpus frequencies (not a run's train_fraction subset): this
    # builds targets for new runs, so it should see all the data.
    freq = load_corpus_freq(args.train_repo)
    rows = load_unique_rows(args.train_repo)
    print(f"{len(rows)} unique rows")
    if args.exclude_calls_from:
        excluded = set()
        for path in args.exclude_calls_from.split(","):
            excluded |= {json.loads(line)["call_id"] for line in open(path, encoding="utf-8")}
        rows = [r for r in rows if r["call_id"] not in excluded]
        print(f"{len(rows)} rows left after excluding {len(excluded)} calls from {args.exclude_calls_from}")
    if args.max_rows and args.max_rows < len(rows):
        rows = random.Random(args.seed).sample(rows, args.max_rows)

    strata = Counter()
    n_diffs = n_rule_edits = 0
    with Pool(args.workers, initializer=_init, initargs=(freq,)) as pool, \
            open(out / "rows.jsonl", "w", encoding="utf-8") as rf, \
            open(out / "diffs.jsonl", "w", encoding="utf-8") as df:
        for row_out, diffs in tqdm(pool.imap(process_row, rows, chunksize=64), total=len(rows)):
            rf.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            for d in diffs:
                df.write(json.dumps(d, ensure_ascii=False) + "\n")
            strata.update(d["stratum"] for d in diffs)
            n_diffs += len(diffs)
            n_rule_edits += row_out["rule_edits"]

    stats = {"rows": len(rows), "rule_edits": n_rule_edits, "diffs": n_diffs, "by_stratum": dict(strata),
             "judgeable": n_diffs - sum(strata[k] for k in NOT_JUDGED)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
