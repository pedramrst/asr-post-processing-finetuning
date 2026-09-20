"""YAML config loading for the SFT training script.

The YAML file groups related hyperparameters into sections (`training`,
`lora`, `quantization`, `hub`, `tensorboard`) purely for readability; they are
flattened into a single `Config` dataclass that `train.py` consumes. Unknown
keys raise an error immediately rather than being silently ignored, since a
typo in a sweep config (e.g. `laarning_rate`) would otherwise waste a full
training run before anyone notices.
"""
from __future__ import annotations

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
    input_column: str = "text_whisper"
    target_column: str = "text"
    system_prompt: str | None = None

    output_dir: str = "./outputs/run"
    max_length: int = 512
    # "drop" (default) excludes rows whose tokenized prompt+target exceeds
    # max_length, rather than truncating into the target and training on an
    # incomplete correction. "truncate" keeps them but cuts off the end of
    # the target. A row is always dropped regardless of this setting if the
    # prompt alone exceeds max_length (no room left for any target token).
    on_long_example: str = "drop"

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
    test_max_examples: int | None = None


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
    """
    if not isinstance(value, str):
        return value
    if typing.get_origin(target_type) is typing.Union:
        non_none = [a for a in typing.get_args(target_type) if a is not type(None)]
        if len(non_none) != 1:
            return value  # ambiguous union (e.g. bool | str) -- leave as given
        target_type = non_none[0]
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
