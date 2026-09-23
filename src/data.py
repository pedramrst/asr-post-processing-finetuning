"""Dataset loading and tokenization for the Whisper-output correction SFT task."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jiwer
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a post-editor for a Persian call-center speech-to-text system. You "
    "will receive a raw transcript from an ASR model. General vocabulary is "
    "usually already correct -- the ASR model mainly struggles with named "
    "entities: person names, place names, and product or order names, since "
    "these are exactly the words it is least equipped to recognize. Pay "
    "special attention to any name-like word: if it looks like a plausible "
    "mishearing of a real name (a phonetically similar but different word "
    "given the surrounding context), correct it to the most likely intended "
    "name; if it already looks right, leave it as is. Elsewhere, fix other "
    "mis-heard words, remove any repeated or clearly hallucinated fragments, "
    "and use correct Persian ZWNJ half-spacing for compound words. Do not add "
    "punctuation, and write numbers as words, not digits. If the transcript is "
    "already correct, return it unchanged. Output only the corrected transcript "
    "-- no explanation, no extra text."
)

# Same task, but for training/eval against a punctuated target (e.g. text_soniox
# in our own curated data, text_raw on the ErfanRou/callcc-test-1k Hub dataset)
# instead of the default punctuation-free text -- see Config.include_punctuation.
# Whisper's own output (text_whisper, this model's actual input) never has
# punctuation, so this is a strictly harder version of the task: no acoustic
# pause/prosody cues survive into the text the model actually sees, unlike
# whatever produced the punctuated reference in the first place.
SYSTEM_PROMPT_WITH_PUNCTUATION = SYSTEM_PROMPT.replace(
    "and use correct Persian ZWNJ half-spacing for compound words. Do not add "
    "punctuation, and write numbers as words, not digits.",
    "use correct Persian ZWNJ half-spacing for compound words, and add "
    "appropriate Persian punctuation (commas, periods, question marks) where "
    "the sentence structure calls for it, but write numbers as words, not digits.",
)


def resolve_system_prompt(cfg) -> str:
    """The system prompt actually used for a run: cfg.system_prompt if set
    explicitly, else the punctuation-aware or punctuation-free default
    depending on cfg.include_punctuation. Centralized so every call site
    (training's tokenization, baseline/checkpoint/final eval, the entity/typo
    secondary evals) picks the same prompt for the same config -- a run
    training on a punctuated target but evaluating with the punctuation-free
    prompt (or vice versa) would silently teach/measure the wrong thing.
    """
    if cfg.system_prompt:
        return cfg.system_prompt
    return SYSTEM_PROMPT_WITH_PUNCTUATION if cfg.include_punctuation else SYSTEM_PROMPT


def load_local_jsonl_columns(path: str, columns: list[str]) -> Dataset:
    """Reads a local .jsonl file, keeping only `columns`.

    `datasets.load_dataset("json", ...)` loads every column and asks pyarrow
    to infer one unified schema for the whole file. build_dataset.py's output
    has a `crm_context` column that's arbitrary, inconsistently-shaped JSON
    copied straight from CRM records (a dict with no fixed set of keys, or
    null) -- across chunks that confuses pyarrow's schema inference and the
    load fails with e.g. "Couldn't cast array of type string to null", even
    though training only ever reads two flat string columns. Parsing lines by
    hand and dropping everything else sidesteps that entirely.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({c: obj.get(c) for c in columns})
    return Dataset.from_list(rows)


def load_sft_dataset(
    dataset_id: str,
    train_split: str,
    eval_split: str | None,
    eval_fraction: float | None = None,
    seed: int = 42,
    input_column: str = "text_whisper",
    target_column: str = "text",
    train_fraction: float | None = None,
) -> DatasetDict:
    """Load the already-preprocessed dataset, from the Hub or a local file.

    This assumes the input/target columns are already cleaned and filtered
    upstream (e.g. by build_dataset.py) — swap this out if your teammate's
    loading step does anything beyond that.

    `dataset_id` pointing at an existing local path (as produced by
    build_dataset.py, a single unsplit .jsonl) is loaded as
    JSON Lines and, since it has no predefined splits, `eval_fraction` carves
    out a deterministic held-out slice for validation instead of `eval_split`
    (which only makes sense for a Hub dataset with named splits).

    `train_fraction`, if set, deterministically (via `seed`) subsamples only
    the train split afterwards -- e.g. for a data-scaling ablation (train on
    10%/25%/50%/100% of the curated data, compare eval_loss/test WER) without
    needing a separate curated file per fraction. `eval`/`validation` is left
    at full size either way, since shrinking it would make those comparisons
    noisier, not cheaper in any way that matters.
    """
    if Path(dataset_id).exists():
        raw = load_local_jsonl_columns(dataset_id, [input_column, target_column])
        if eval_fraction:
            split = raw.train_test_split(test_size=eval_fraction, seed=seed)
            result = DatasetDict(train=split["train"], validation=split["test"])
        else:
            result = DatasetDict(train=raw)
    else:
        train_ds = load_dataset(dataset_id, split=train_split)
        if eval_split:
            eval_ds = load_dataset(dataset_id, split=eval_split)
            result = DatasetDict(train=train_ds, validation=eval_ds)
        else:
            result = DatasetDict(train=train_ds)

    if train_fraction is not None and train_fraction < 1.0:
        n = round(len(result["train"]) * train_fraction)
        result["train"] = result["train"].shuffle(seed=seed).select(range(n))

    return result


# Persian letter groups that are true homophones -- an artifact of spelling
# preserving Arabic-loanword distinctions that collapsed to one sound in
# Persian pronunciation (e.g. ز/ذ/ض/ظ are all just /z/). Mapping each group
# to one canonical letter before comparing is what actually separates a
# realistic ASR confusion (زفری -> ظفری, homophone substitution) from a word
# that just happens to share some letters (الویزی -> پرویزی) -- tested
# directly: plain character-set overlap and unnormalized edit distance both
# scored the homophone case *lower* than at least one genuinely unrelated
# pair, because Persian names often share a common suffix (e.g. "...ویزی")
# that inflates either metric regardless of whether the actual word is
# recoverable. ک/گ are deliberately excluded -- they're distinct phonemes in
# Persian, not homophones, even though they look and sound similar-ish.
_PHONETIC_GROUPS = ["زذضظ", "سصث", "تط", "قغ", "حه"]
_PHONETIC_NORMALIZE = {ch: group[0] for group in _PHONETIC_GROUPS for ch in group}


def _phonetic_similarity(a: str, b: str) -> float:
    """1 - normalized character-level edit distance between `a` and `b`,
    after ZWNJ-stripping and Persian-homophone normalization (see
    _PHONETIC_GROUPS). 1.0 means identical once homophone spelling
    differences are accounted for; 0.0 means completely different.
    """
    def normalize(s: str) -> str:
        s = s.replace("‌", "")
        return "".join(_PHONETIC_NORMALIZE.get(ch, ch) for ch in s)

    a2, b2 = normalize(a), normalize(b)
    if not a2 and not b2:
        return 1.0
    edits = jiwer.process_characters([a2], [b2])
    total_edits = edits.substitutions + edits.deletions + edits.insertions
    return 1 - total_edits / max(len(a2), len(b2))


def low_signal_word_spans(
    whisper_text: str,
    target_text: str,
    min_similarity: float = 0.75,
    corpus_freq: Counter | None = None,
    max_common_freq: int = 1,
) -> list[tuple[int, int]]:
    """Character [start, end) spans in `target_text` where correcting
    whisper_text has no recoverable signal to learn from -- see
    Config.mask_low_signal_corrections and README's "Low-signal correction
    masking". Thin wrapper around low_signal_word_ranges() (see its
    docstring for what "no recoverable signal" means and the three signals
    that can rescue a word from it) that converts its word-index ranges to
    character offsets, for callers that need to slice/mask `target_text`
    directly (here, token-level loss masking via the offset mapping).

    Verified directly on a sample of confirmed entity corrections: roughly
    half had no recoverable signal at all -- the target word bears little
    or no phonetic resemblance to what Whisper actually produced (e.g.
    "الویزی" -> "پرویزی", or a word Whisper produced nothing for at all) --
    while ones like "زفری" -> "ظفری" do (ز/ظ are true Persian homophones).
    Training on the unrecoverable ones the same as everything else teaches
    confident-sounding guessing, not real correction, so this flags them for
    the caller to exclude from the training loss instead.

    `min_similarity` is checked with _phonetic_similarity(), not plain
    character overlap -- tested directly: unnormalized metrics (character-set
    overlap, raw edit distance) scored the homophone case *lower* than a
    genuinely unrelated pair, since Persian names often share a common
    suffix that inflates similarity regardless of whether the word is
    actually recoverable. The default 0.75 was picked to cleanly separate
    known guessable pairs (>= 0.833: homophone substitutions, ZWNJ/spacing-
    only differences) from known unguessable ones (<= 0.714) on that sample
    -- like any threshold tuned on a small sample, treat it as a reasonable
    starting point to revisit with more data, not a precise cutoff.

    `corpus_freq` (a word -> corpus-wide occurrence count, e.g. from
    train.py's Counter over every training-target text) adds a second,
    necessary gate: verified directly against real training data, phonetic
    similarity alone flagged mostly *common function/filler words*
    ("رو", "بله", "و", "هم", "خب"), not names -- a "delete" chunk (Whisper
    produced nothing at all) always scores 0.0 similarity by construction,
    so a dropped "بله" (yes) was masked exactly as often as a dropped name,
    even though "بله" is trivially predictable from Persian dialogue
    structure regardless of what Whisper produced, unlike an arbitrary
    proper noun. `corpus_freq=None` (the default) skips this gate entirely,
    e.g. for testing this function in isolation without a corpus.
    """
    ranges = low_signal_word_ranges(whisper_text, target_text, min_similarity, corpus_freq, max_common_freq)
    return _word_ranges_to_char_spans(target_text, ranges)


def _word_ranges_to_char_spans(target_text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Converts inclusive [first, last] *word*-index ranges (as returned by
    low_signal_word_ranges()/recoverable_correction_word_ranges()) to
    character [start, end) spans in `target_text`, for callers that need to
    slice/mask/tag the text directly."""
    if not ranges:
        return []
    target_words = target_text.split()

    # Character start offset of each target word, in order -- target_text is
    # built from " ".join(...) upstream (build_dataset.py), so words are
    # single-space-separated and this simple forward scan is exact.
    word_starts = []
    pos = 0
    for w in target_words:
        idx = target_text.index(w, pos)
        word_starts.append(idx)
        pos = idx + len(w)

    return [(word_starts[first], word_starts[last] + len(target_words[last])) for first, last in ranges]


def _is_common_enough(word: str, corpus_freq: Counter, max_common_freq: int) -> bool:
    """True if `word` -- or, when it's ZWNJ-joined, every one of its
    ZWNJ-split parts -- occurs more than `max_common_freq` times in
    `corpus_freq`.

    Handles words that are only rare because of an inconsistently-applied
    half-space, not because they're actually uncommon: verified directly,
    "هیچ‌جایش" (joined) occurs once in the training corpus, but its parts
    split apart -- "هیچ" (33,613) and "جایش" (14) -- are both individually
    well above the gate. Checked across every rare (<= max_common_freq)
    ZWNJ-joined word in the corpus: 83.3% flip to "common enough" once
    looked up this way. This only affects the *lookup* used for this
    recoverability decision -- the word itself isn't rewritten anywhere
    else in the pipeline (tokenization, alignment, WER all stay untouched),
    so this doesn't carry the risk a global ZWNJ-stripping pass would (that
    would also un-join real compounds like "می‌کنم", changing word counts
    and alignment indices everywhere they're used)."""
    if corpus_freq.get(word, 0) > max_common_freq:
        return True
    if "‌" in word:
        parts = word.split("‌")
        if all(corpus_freq.get(p, 0) > max_common_freq for p in parts):
            return True
    return False


def low_signal_word_ranges(
    whisper_text: str,
    target_text: str,
    min_similarity: float = 0.75,
    corpus_freq: Counter | None = None,
    max_common_freq: int = 1,
) -> list[tuple[int, int]]:
    """Inclusive [first, last] *word*-index ranges in `target_text` with no
    recoverable correction signal -- the shared core of
    low_signal_word_spans() (which converts these to character spans for
    token masking) and evaluate.py's fix_rate_recoverable (which drops these
    positions from the denominator, so it measures corrections the model
    could actually have made rather than counting impossible ones as
    failures). See low_signal_word_spans' docstring for what "no recoverable
    signal" means and why each gate is there.

    Decided per WORD, not per jiwer alignment chunk: a multi-word
    substitute/delete chunk used to be masked (or not) as one atomic unit,
    which meant a single genuinely-rare word could drag ordinary neighbors
    down with it -- verified directly on real data: the 3-word span
    "هیچ‌جایش جزییات سفارشتون" was masked entirely because "هیچ‌جایش" (freq
    1) failed the gate, even though "جزییات" (2,121) and "سفارشتون" (8,604)
    are both individually common and easily guessable on their own. Each
    reference word in a chunk is now judged independently, so every range
    returned here is always a single word ((i, i)), never a merged span --
    functionally equivalent for the two callers above (both just need to
    know which character/token positions to exclude), so this doesn't
    change their behavior beyond making the exclusions more precise.

    A word is rescued from masking (kept in the loss / scored) by any ONE
    of three independent signals:
      1. Phonetically similar enough to what Whisper actually produced at
         this position (checked against the whole chunk's Whisper span,
         since jiwer doesn't give a finer sub-alignment within one chunk).
      2. Common enough to be guessable from context regardless of what
         Whisper said -- see _is_common_enough().
      3. Also appears elsewhere in THIS row's own Whisper output --
         Whisper independently produced the same word correctly somewhere
         else in the same call (only possible for assembled/whole-call
         rows, not single-segment chunked ones), which is a real in-context
         clue the model can learn to use, not just corpus-wide commonness.
         Verified directly: 27.7% of words masked under signals 1+2 alone
         also satisfy this. Checked as plain membership in the row's
         Whisper word set, not requiring that other occurrence to be a
         confirmed-correct jiwer alignment at its own position -- measured
         directly that the stricter, alignment-based version agrees with
         this simpler one 99.5% of the time (207/208) once this function's
         per-word decision (the change above) is applied, so the extra
         complexity of tracking per-occurrence alignments isn't worth it
         for an ASR model that's confirmed not to hallucinate content.
    """
    categories = word_correction_categories(whisper_text, target_text, min_similarity, corpus_freq, max_common_freq)
    return [(i, i) for i, c in enumerate(categories) if c == 3]


def word_correction_categories(
    whisper_text: str,
    target_text: str,
    min_similarity: float = 0.75,
    corpus_freq: Counter | None = None,
    max_common_freq: int = 1,
) -> list[int]:
    """Per-target-word category, one entry per `target_text.split()` word:

      1 -- Whisper already had this right; no correction needed.
      2 -- Whisper got this wrong, but recoverable (any one of
           low_signal_word_ranges' three rescue signals applies) -- a
           genuine correction with real signal to learn from.
      3 -- Whisper got this wrong and it's NOT recoverable -- exactly the
           word positions low_signal_word_ranges() returns.

    Shared core of low_signal_word_ranges() (uses category 3, for
    Config.mask_low_signal_corrections/fix_rate_recoverable) and
    Config.recoverable_correction_weight (uses category 2, up-weighted in
    the training loss) -- computed together in one function so the two
    features can't silently disagree about which word is which category
    from drifting, separately-maintained implementations.
    """
    target_words = target_text.split()
    whisper_words = whisper_text.split()
    if not target_words:
        return []
    whisper_word_set = set(whisper_words)
    alignment = jiwer.process_words([target_text], [whisper_text]).alignments[0]

    categories = [1] * len(target_words)
    for chunk in alignment:
        if chunk.type not in ("substitute", "delete"):
            continue  # "equal" stays category 1; "insert" has no target-side word
        ref_span_words = target_words[chunk.ref_start_idx : chunk.ref_end_idx]
        hyp_span = " ".join(whisper_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
        for offset, w in enumerate(ref_span_words):
            idx = chunk.ref_start_idx + offset
            if _phonetic_similarity(w, hyp_span) >= min_similarity:
                categories[idx] = 2  # guessable from what Whisper actually said here
            elif corpus_freq is not None and _is_common_enough(w, corpus_freq, max_common_freq):
                categories[idx] = 2  # common enough to be guessable from context alone
            elif w in whisper_word_set:
                categories[idx] = 2  # Whisper independently got this right elsewhere in this call
            else:
                categories[idx] = 3  # unrecoverable
    return categories


def recoverable_correction_word_ranges(
    whisper_text: str,
    target_text: str,
    min_similarity: float = 0.75,
    corpus_freq: Counter | None = None,
    max_common_freq: int = 1,
) -> list[tuple[int, int]]:
    """Inclusive [first, last] *word*-index ranges in `target_text` that are
    genuine, recoverable corrections -- category 2 of
    word_correction_categories() (see its docstring). Counterpart to
    low_signal_word_ranges() (category 3): together the two partition every
    substitute/delete word into "worth training on as-is" (this one) vs.
    "worth excluding from the loss" (that one); every "equal" word (Whisper
    already had it right) is in neither.

    Feeds Config.recoverable_correction_weight -- see
    _weight_recoverable_correction_spans() -- to up-weight these positions
    in the training loss, on the theory that they're diluted by far more
    already-correct passthrough tokens in the same example."""
    categories = word_correction_categories(whisper_text, target_text, min_similarity, corpus_freq, max_common_freq)
    return [(i, i) for i, c in enumerate(categories) if c == 2]


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Start index of the first contiguous occurrence of `needle` in
    `haystack`, or -1 if it doesn't occur. Sizes here are small (one
    example's token ids), so the naive scan is fine."""
    if not needle:
        return -1
    first = needle[0]
    limit = len(haystack) - len(needle)
    for i in range(limit + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _locate_target_tokens(
    tokenizer: PreTrainedTokenizerBase,
    target_text: str,
    input_ids: list[int],
    prompt_len: int,
) -> tuple[list[tuple[int, int]], int] | None:
    """Tokenizes `target_text` standalone and locates its token ids as a
    contiguous subsequence within `input_ids[prompt_len:]` -- NOT assumed to
    start exactly at `prompt_len`, since some chat templates insert
    boilerplate between the prompt and the actual assistant content (e.g.
    Qwen3's chat template adds an empty "<think>\\n\\n</think>\\n\\n" block
    there even when the message itself has no such content). Shared prep
    step for _mask_low_signal_spans()/_weight_recoverable_correction_spans()
    -- both need to map `target_text`-relative character spans to token
    positions in the same tokenized example.

    `input_ids` must be the example's actual, unmodified token ids (e.g.
    `kept_ids`, NOT `labels`) -- searching within `labels` would break if
    another low-signal pass already ran first and wrote -100 into part of
    the region being searched, since -100 is never a real token id and the
    subsequence would no longer match.

    Returns (target_offsets, start) -- `target_offsets[i]`'s corresponding
    position in `input_ids` is `start + i` -- or None if the subsequence
    isn't found at all, e.g. a row truncated mid-target (callers should
    treat that as "nothing to mask/weight", not an error).
    """
    target_enc = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    target_ids = target_enc["input_ids"]
    offset = _find_subsequence(input_ids[prompt_len:], target_ids)
    if offset == -1:
        return None
    return target_enc["offset_mapping"], prompt_len + offset


def _mask_low_signal_spans(
    tokenizer: PreTrainedTokenizerBase,
    target_text: str,
    whisper_text: str,
    input_ids: list[int],
    labels: list[int],
    prompt_len: int,
    min_similarity: float,
    corpus_freq: Counter | None,
    max_common_freq: int,
) -> int:
    """Sets labels[i] = -100 for target tokens that fall inside a low-signal
    correction span (see low_signal_word_spans) -- corrections that aren't
    guessable from context, which we don't want to train the model to
    reproduce by memorization or, worse, teach it to guess at randomly (this
    is a property of Whisper's failures on that word, not something we want
    the model to learn to do).

    Maps low_signal_word_spans' character spans (computed over target_text)
    to token positions via _locate_target_tokens() (see its docstring).
    Falls back to no masking (safe/conservative) if the target's tokens
    aren't found in `input_ids` at all, e.g. a row truncated mid-target.

    Returns the number of tokens masked this way.
    """
    spans = low_signal_word_spans(whisper_text, target_text, min_similarity, corpus_freq, max_common_freq)
    if not spans:
        return 0

    located = _locate_target_tokens(tokenizer, target_text, input_ids, prompt_len)
    if located is None:
        return 0
    target_offsets, start = located

    masked = 0
    for i, (char_start, char_end) in enumerate(target_offsets):
        if char_start == char_end:
            continue
        if any(char_start < span_end and char_end > span_start for span_start, span_end in spans):
            labels[start + i] = -100
            masked += 1
    return masked


def _weight_recoverable_correction_spans(
    tokenizer: PreTrainedTokenizerBase,
    target_text: str,
    whisper_text: str,
    input_ids: list[int],
    token_weights: list[float],
    prompt_len: int,
    min_similarity: float,
    corpus_freq: Counter | None,
    max_common_freq: int,
    weight: float,
) -> int:
    """Sets token_weights[i] = `weight` for target tokens that fall inside a
    genuine, recoverable correction (see recoverable_correction_word_ranges)
    -- up-weights the loss on words Whisper got wrong but that have real
    signal to learn from, on the theory that they're a small minority of
    tokens in any given example (most of the target is already-correct
    passthrough text) and so contribute little to the gradient relative to
    how much we actually want the model to learn from them.

    Same token-location mechanism as _mask_low_signal_spans() (see
    _locate_target_tokens()) -- category 2 and category 3
    (word_correction_categories()) are disjoint by construction, so a token
    weighted here is never also one _mask_low_signal_spans() masks, and vice
    versa.

    Returns the number of tokens weighted this way.
    """
    ranges = recoverable_correction_word_ranges(whisper_text, target_text, min_similarity, corpus_freq, max_common_freq)
    spans = _word_ranges_to_char_spans(target_text, ranges)
    if not spans:
        return 0

    located = _locate_target_tokens(tokenizer, target_text, input_ids, prompt_len)
    if located is None:
        return 0
    target_offsets, start = located

    weighted = 0
    for i, (char_start, char_end) in enumerate(target_offsets):
        if char_start == char_end:
            continue
        if any(char_start < span_end and char_end > span_start for span_start, span_end in spans):
            token_weights[start + i] = weight
            weighted += 1
    return weighted


def build_example(
    tokenizer: PreTrainedTokenizerBase,
    whisper_text: str,
    target_text: str,
    max_length: int,
    system_prompt: str = SYSTEM_PROMPT,
    on_long_example: str = "drop",
    mask_low_signal_corrections: bool = False,
    mask_min_similarity: float = 0.75,
    mask_corpus_freq: Counter | None = None,
    mask_max_common_freq: int = 1,
    recoverable_correction_weight: float = 1.0,
) -> dict:
    """Tokenize one (whisper_text -> target_text) pair with loss masked to the target span.

    We render the prompt and the full conversation separately and diff their
    token lengths, rather than relying on `apply_chat_template(...,
    return_assistant_tokens_mask=True)`: none of the candidate models' chat
    templates define the `{% generation %}` block that feature needs, so it
    silently returns an all-zero mask (no error) instead of failing loudly.

    Returns a dict with input_ids/attention_mask/labels/token_weights plus a
    `status` key ("ok", "truncated", or "dropped") -- callers should filter
    out "dropped" rows (train.py does this after `.map()`; every row must
    return the same schema during the map itself, so a dropped row still
    gets placeholder tensor fields) -- a `masked_low_signal_tokens` count
    (always 0 unless `mask_low_signal_corrections` is set), and a
    `weighted_recoverable_tokens` count (always 0 unless
    `recoverable_correction_weight != 1.0`).

    `on_long_example` controls what happens when the full prompt+target
    exceeds `max_length`:
      - "drop" (default): exclude the row entirely, rather than truncating
        into the *target* and training the model to produce a correction
        that just stops mid-sentence.
      - "truncate": keep it, cutting off the end of the target.
    A row is always dropped regardless of this setting if the prompt alone
    already exceeds max_length -- truncating would leave zero target tokens
    to compute loss on, i.e. an example with no training signal at all.

    `mask_low_signal_corrections`, if set, additionally masks out (labels =
    -100) the target tokens for any word-level correction that
    low_signal_word_spans() flags as not recoverable from context (see its
    docstring) -- so the model isn't trained to reproduce, or implicitly
    rewarded/penalized for guessing at, a correction with no real signal in
    the input. Everything else in the target -- correct passthrough text and
    guessable corrections alike -- still trains normally. `mask_corpus_freq`/
    `mask_max_common_freq` are passed straight through to
    low_signal_word_spans() as its frequency gate -- without it, phonetic
    similarity alone flags mostly common function/filler words, not names
    (see that function's docstring).

    `recoverable_correction_weight`, if not 1.0, multiplies the per-token
    loss (see train.py's WeightedLossTrainer) for target tokens inside a
    genuine, recoverable correction -- see
    recoverable_correction_word_ranges()/Config.recoverable_correction_weight
    -- on the theory that they're a small minority of any example's tokens
    (most of the target is already-correct passthrough text or, with
    masking on, excluded entirely) and so get diluted in the gradient
    relative to how much we actually want the model to learn from them.
    Reuses `mask_corpus_freq`/`mask_min_similarity`/`mask_max_common_freq` as
    its own signals too -- both features are answering "is this word
    guessable," just keeping opposite halves of the same answer (see
    word_correction_categories()), so they share one set of thresholds
    rather than risking two configs disagreeing about the same word.
    """
    if on_long_example not in ("drop", "truncate"):
        raise ValueError(f"on_long_example must be 'drop' or 'truncate', got {on_long_example!r}")

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": whisper_text},
    ]
    prompt_str = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_messages = prompt_messages + [{"role": "assistant", "content": target_text}]
    full_str = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt is not a token-level prefix of the full sequence for this "
            "tokenizer; the label-masking assumption in build_example() does not "
            "hold for this model and needs a different masking strategy."
        )

    if len(prompt_ids) >= max_length:
        status = "dropped"
    elif len(full_ids) > max_length:
        status = "dropped" if on_long_example == "drop" else "truncated"
    else:
        status = "ok"

    kept_ids = full_ids[:max_length]
    labels = list(kept_ids)
    mask_len = min(len(prompt_ids), len(kept_ids))
    labels[:mask_len] = [-100] * mask_len
    token_weights = [1.0] * len(kept_ids)

    masked_low_signal_tokens = 0
    if mask_low_signal_corrections and status != "dropped":
        masked_low_signal_tokens = _mask_low_signal_spans(
            tokenizer, target_text, whisper_text, kept_ids, labels, len(prompt_ids), mask_min_similarity,
            mask_corpus_freq, mask_max_common_freq,
        )

    weighted_recoverable_tokens = 0
    if recoverable_correction_weight != 1.0 and status != "dropped":
        weighted_recoverable_tokens = _weight_recoverable_correction_spans(
            tokenizer, target_text, whisper_text, kept_ids, token_weights, len(prompt_ids), mask_min_similarity,
            mask_corpus_freq, mask_max_common_freq, recoverable_correction_weight,
        )

    return {
        "input_ids": kept_ids,
        "attention_mask": [1] * len(kept_ids),
        "labels": labels,
        "token_weights": token_weights,
        "status": status,
        "masked_low_signal_tokens": masked_low_signal_tokens,
        "weighted_recoverable_tokens": weighted_recoverable_tokens,
    }


@dataclass
class PadCollator:
    """Pads input_ids/attention_mask/labels(/token_weights) to the longest
    sequence in the batch.

    `include_token_weights` defaults off -- matching Config.
    recoverable_correction_weight's own default of 1.0 (a no-op) -- so a run
    that doesn't use that feature gets a batch dict with exactly the same
    keys as before it existed, and train.py can keep using the plain
    `Trainer` (not `WeightedLossTrainer`) for it, untouched by any of this."""

    pad_token_id: int
    label_pad_id: int = -100
    include_token_weights: bool = False

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels, token_weights = [], [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_id] * pad_len)
            if self.include_token_weights:
                # Padding value is irrelevant -- WeightedLossTrainer only
                # ever applies weights to positions with a real label
                # (!= -100), and padding is always labeled -100 above.
                token_weights.append(f["token_weights"] + [1.0] * pad_len)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.include_token_weights:
            batch["token_weights"] = torch.tensor(token_weights, dtype=torch.float32)
        return batch
