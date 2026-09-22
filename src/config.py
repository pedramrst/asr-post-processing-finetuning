"""YAML config loading for the SFT training script.

The YAML file groups related hyperparameters into sections (`training`,
`lora`, `quantization`, `hub`, `tensorboard`) purely for readability; they are
flattened into a single `Config` dataclass that `train.py` consumes. Unknown
keys raise an error immediately rather than being silently ignored, since a
typo in a sweep config (e.g. `laarning_rate`) would otherwise waste a full
training run before anyone notices.
"""
from __future__ import annotations

import types
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# Maps a YAML section name to either:
#   - a set of allowed sub-keys that map 1:1 onto Config field names, or
#   - a dict mapping sub-key -> Config field name, when they differ.
_SECTIONS: dict[str, set[str] | dict[str, str]] = {
    "training": {
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_ratio",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "seed",
    },
    "lora": {"r": "lora_r", "alpha": "lora_alpha", "dropout": "lora_dropout"},
    "quantization": {"load_in_4bit"},
    "hub": {
        "push_to_hub": "push_to_hub",
        "repo_id": "hub_repo_id",
        "folder": "hub_repo_folder",
        "private": "hub_private",
    },
    "tensorboard": {"logging_dir": "tensorboard_logging_dir"},
    "test": {
        "dataset_id": "test_dataset_id",
        "split": "test_split",
        "input_column": "test_input_column",
        "target_column": "test_target_column",
        "max_new_tokens": "test_max_new_tokens",
        "batch_size": "test_batch_size",
        "max_examples": "test_max_examples",
        "checkpoint_max_examples": "test_checkpoint_max_examples",
        "hallucination_overlap_floor": "test_hallucination_overlap_floor",
        "repetition_penalty": "test_repetition_penalty",
        "no_repeat_ngram_size": "test_no_repeat_ngram_size",
        "baseline": "test_baseline",
        "entity_dataset_id": "test_entity_dataset_id",
        "typo_dataset_id": "test_typo_dataset_id",
    },
}


@dataclass
class Config:
    model_id: str
    dataset_id: str
    train_split: str = "train"
    eval_split: str | None = None
    # Only used when dataset_id is a local file with no predefined splits
    # (e.g. build_dataset.py's output): fraction of it held
    # out for validation. Ignored for a Hub dataset_id -- use eval_split there.
    eval_fraction: float | None = None
    # Deterministically subsamples only the train split to this fraction of
    # its rows (e.g. 0.1 for 10%) -- for a data-scaling ablation without a
    # separate curated file per fraction. Leaves eval/validation full-size.
    # None (default) or 1.0 uses every row.
    train_fraction: float | None = None
    input_column: str = "text_whisper"
    target_column: str = "text"
    system_prompt: str | None = None
    # Feature flag for training/eval against a punctuated target instead of
    # the default punctuation-free `text` -- see data.py's
    # SYSTEM_PROMPT_WITH_PUNCTUATION and README's "Punctuation" section. Only
    # swaps the system prompt (unless system_prompt is set explicitly, which
    # always wins) -- it does NOT change target_column/test_target_column,
    # since the punctuated column has a different name depending on the
    # dataset (text_soniox in our own curated data, text_raw on
    # ErfanRou/callcc-test-1k and the entity/typo eval slices); set those
    # explicitly alongside this. train.py warns at startup if this and
    # target_column look inconsistent with each other.
    include_punctuation: bool = False

    output_dir: str = "./outputs/run"
    max_length: int = 512
    # "drop" (default) excludes rows whose tokenized prompt+target exceeds
    # max_length, rather than truncating into the target and training on an
    # incomplete correction. "truncate" keeps them but cuts off the end of
    # the target. A row is always dropped regardless of this setting if the
    # prompt alone exceeds max_length (no room left for any target token).
    on_long_example: str = "drop"
    # Masks (labels = -100) target tokens for word-level corrections that
    # aren't recoverable from what Whisper actually produced -- see
    # data.py's low_signal_word_spans()/build_example() and README's
    # "Low-signal correction masking" section. Off by default: opt in to
    # test it against the baseline of training on every corrected word.
    mask_low_signal_corrections: bool = False
    # Phonetic/character similarity threshold (0-1, higher = stricter) below
    # which a correction is masked out -- see low_signal_word_spans().
    mask_min_similarity: float = 0.75
    # A phonetically-dissimilar correction is only masked if at least one of
    # its words occurs at most this many times across the whole training
    # set (train.py builds this count once before tokenizing) -- without
    # this gate, phonetic similarity alone flags mostly common function/
    # filler words ("رو", "بله", "و"), not names, since a word Whisper
    # produced nothing for always scores 0 similarity by construction
    # regardless of how common it is. See low_signal_word_spans().
    mask_max_common_freq: int = 1

    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int | None = 3
    seed: int = 42

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    load_in_4bit: bool = False
    # Loads the base model + sets up LoRA via Unsloth's FastLanguageModel
    # instead of plain transformers.AutoModelForCausalLM + peft.get_peft_model
    # -- Unsloth patches the model with fused kernels and its own
    # gradient-checkpointing implementation for reported ~2x faster training
    # and ~70% less VRAM on Qwen3/Gemma3, without changing anything else
    # about this script (still a plain transformers.Trainer underneath, not
    # TRL's SFTTrainer -- Unsloth's optimization is applied at model-loading
    # time, so it works with any trainer). Requires a CUDA GPU and the
    # `unsloth` package -- NOT verified end-to-end in this repo (no CUDA
    # available in the dev environment this was written in); test on a real
    # GPU before relying on it for a full run.
    use_unsloth: bool = False

    # All runs pushing to the same hub_repo_id share ONE Hub repo, each in its
    # own folder -- hub_repo_folder null defaults to output_dir's basename.
    # Every checkpoint save re-syncs output_dir into that folder (full
    # resumable state, stale local-pruned checkpoints deleted remotely too).
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_repo_folder: str | None = None
    hub_private: bool = True

    tensorboard_logging_dir: str | None = None

    # false | true (auto-resume from the latest local checkpoint in output_dir) |
    # a local checkpoint path | "hub" to pull this run's folder from hub_repo_id first.
    resume_from_checkpoint: bool | str = False

    # If set, the model is run against this test set at every checkpoint save
    # (results overwrite test_eval/predictions.jsonl + metrics.json under
    # output_dir with the latest state; the aggregate WER is also logged to
    # TensorBoard each time so you get a WER-vs-step curve, not just a final
    # number). Leave unset to skip test evaluation entirely.
    test_dataset_id: str | None = None
    test_split: str = "test"
    test_input_column: str = "text_whisper"
    test_target_column: str = "text"
    test_max_new_tokens: int = 256
    test_batch_size: int = 8
    # Full-set size used for the one-time baseline (before training) and
    # final (after training) evals. Leave null for the full test set -- those
    # only run once each, so the accurate number is worth the time.
    test_max_examples: int | None = None
    # Separate, usually much smaller cap for the *repeated* per-checkpoint
    # eval (TestEvalCallback) -- with eval_steps=100 over a long run, that
    # can fire hundreds of times, so running the full set there every time
    # multiplies eval cost far beyond training itself. null falls back to
    # test_max_examples (i.e. no separate cap).
    test_checkpoint_max_examples: int | None = None
    # A prediction is flagged "hallucinated" (predictions.jsonl + the
    # test_hallucination_rate TensorBoard curve) when under this % of its
    # words appear anywhere in the input -- i.e. the model said things the
    # input never did, regardless of whether it happens to match the
    # reference. Independent of wer/exact_match, which only compare against
    # the reference and can't tell "wrong correction" from "invented content".
    test_hallucination_overlap_floor: float = 50.0
    # Greedy decoding (test generation uses do_sample=False) is prone to a
    # degenerate failure: once a short, locally-high-probability phrase
    # repeats a couple times -- e.g. this task's real "بله" (yes)
    # agreement-word bursts, which do occur naturally as short runs in
    # training segments -- it can get stuck repeating it for the rest of
    # max_new_tokens instead of stopping. Observed directly: ~1 in 6-7 test
    # outputs on this task's fine-tuned checkpoints degenerated this way,
    # tanking wer/hallucination_rate on otherwise-good predictions. These
    # are the standard mitigation; 1.0/0 disables each respectively.
    test_repetition_penalty: float = 1.2
    test_no_repeat_ngram_size: int = 3
    # Runs one extra test-set eval before the first training step (LoRA's
    # B matrix is zero-initialized, so the freshly-wrapped, untrained model
    # is numerically identical to the plain base model here) -- saved
    # separately under output_dir/test_eval_baseline/ and logged to
    # TensorBoard at step 0, so test_wer/test_exact_match/
    # test_hallucination_rate curves show the pre-fine-tuning starting point,
    # not just the first checkpoint. Skipped on a resumed run (not "before
    # fine-tuning" then) or when test_dataset_id is unset.
    test_baseline: bool = True
    # Optional second, usually much smaller eval set focused on named
    # entities (person/place/order names), run alongside the main test set
    # at baseline/every checkpoint/final -- results saved under
    # test_eval_entity/ (test_eval_entity_baseline/ for the pre-training
    # pass) and logged to TensorBoard as test_entity_wer/
    # test_entity_exact_match/test_entity_hallucination_rate. Aggregate WER
    # on the full test set mixes entity-heavy and entity-free calls
    # together, which can hide exactly the failure mode (weak named-entity
    # correction) this is meant to track on its own. Reuses every other
    # test.* generation setting (input/target column, batch size,
    # repetition_penalty, ...); null skips this entirely.
    test_entity_dataset_id: str | None = None
    # Same idea as test_entity_dataset_id, but for typo/dictation-form
    # errors (stutters, word-boundary merges, ZWNJ half-spacing) instead of
    # named entities -- see build_typo_eval_slice.py. Results under
    # test_eval_typo/ etc., logged as test_typo_wer/... . This is a
    # regression guard, not an improvement target: direct investigation
    # found the model already handles this error category well, so the
    # point of tracking it is catching if pushing harder on entity
    # correction (prepare_split.py's entity upsampling) quietly erodes it.
    test_typo_dataset_id: str | None = None


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        section = _SECTIONS.get(key)
        if section is not None and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key not in section:
                    raise ValueError(f"Unknown config key '{key}.{sub_key}'")
                field_name = section[sub_key] if isinstance(section, dict) else sub_key
                flat[field_name] = sub_value
        else:
            flat[key] = value
    return flat


def _coerce(value: Any, target_type: Any) -> Any:
    """Cast scalar values to their declared field type.

    Needed because PyYAML only parses `1.0e-4`-style scientific notation as a
    float -- `1e-4` (no decimal point) is valid Python but comes back as the
    *string* "1e-4" under YAML 1.1 rules, which would otherwise pass silently
    into TrainingArguments and break there instead of at config-load time.
    Also needed for a `--set some_bool_field=false`-style override: every
    non-empty string is truthy in Python, so without this a bool field would
    silently stay "false" (== True) instead of becoming False.
    """
    if not isinstance(value, str):
        return value
    # `X | None` (PEP 604, used throughout Config's annotations) has origin
    # types.UnionType, a *different* type from typing.Union/Optional[X]'s
    # origin -- both need handling here, or every Optional field silently
    # skips coercion and a `--set` override stays a str.
    if typing.get_origin(target_type) in (typing.Union, types.UnionType):
        non_none = [a for a in typing.get_args(target_type) if a is not type(None)]
        if len(non_none) != 1:
            return value  # ambiguous union (e.g. bool | str) -- leave as given
        target_type = non_none[0]
    if target_type is bool:
        low = value.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse {value!r} as bool -- use true/false.")
    if target_type is float:
        return float(value)
    if target_type is int:
        return int(value)
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    flat = _flatten(raw)
    if overrides:
        flat.update(_flatten(overrides))

    valid_fields = {f.name for f in fields(Config)}
    unknown = set(flat) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config key(s): {sorted(unknown)}")

    type_hints = typing.get_type_hints(Config)
    flat = {key: _coerce(value, type_hints[key]) for key, value in flat.items()}

    return Config(**flat)
