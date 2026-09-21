"""Dataset loading and tokenization for the Whisper-output correction SFT task."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def build_example(
    tokenizer: PreTrainedTokenizerBase,
    whisper_text: str,
    target_text: str,
    max_length: int,
    system_prompt: str = SYSTEM_PROMPT,
    on_long_example: str = "drop",
) -> dict:
    """Tokenize one (whisper_text -> target_text) pair with loss masked to the target span.

    We render the prompt and the full conversation separately and diff their
    token lengths, rather than relying on `apply_chat_template(...,
    return_assistant_tokens_mask=True)`: none of the candidate models' chat
    templates define the `{% generation %}` block that feature needs, so it
    silently returns an all-zero mask (no error) instead of failing loudly.

    Returns a dict with input_ids/attention_mask/labels plus a `status` key
    ("ok", "truncated", or "dropped") -- callers should filter out "dropped"
    rows (train.py does this after `.map()`; every row must return the same
    schema during the map itself, so a dropped row still gets placeholder
    tensor fields).

    `on_long_example` controls what happens when the full prompt+target
    exceeds `max_length`:
      - "drop" (default): exclude the row entirely, rather than truncating
        into the *target* and training the model to produce a correction
        that just stops mid-sentence.
      - "truncate": keep it, cutting off the end of the target.
    A row is always dropped regardless of this setting if the prompt alone
    already exceeds max_length -- truncating would leave zero target tokens
    to compute loss on, i.e. an example with no training signal at all.
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

    return {
        "input_ids": kept_ids,
        "attention_mask": [1] * len(kept_ids),
        "labels": labels,
        "status": status,
    }


@dataclass
class PadCollator:
    """Pads input_ids/attention_mask/labels to the longest sequence in the batch."""

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
