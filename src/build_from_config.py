#!/usr/bin/env python3
"""Resolve or build the ASR-correction training dataset from a data YAML config.

Two modes, chosen by the config's `source` key (see configs/data/default.yaml):

  source: prebuilt -- the dataset is already built, curated, and published to
    a Hub dataset repo (prebuilt.dataset_id). Nothing gets built here; this
    just confirms the repo is reachable and prints what to set
    training.dataset_id to in a train config (configs/train/*.yaml) --
    dataset_id already accepts a Hub id directly, no local file needed.

  source: build -- (re)builds the dataset from raw ErfanRou/callcc-2k via
    build_dataset.py, then curates it via prepare_split.py, using this
    config's `build`/`build.prepare` sections as those scripts' flags. Same
    two scripts as before, just driven from one reviewable/diffable YAML
    file instead of a hand-assembled shell invocation.

Use --set for one-off overrides without editing the file, same convention as
train.py, e.g.:

    python src/build_from_config.py --config configs/data/default.yaml \
        --set source=build --set build.assembled_ratio=0.3

Example:
  python3 src/build_from_config.py --config configs/data/default.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent

# data-YAML key (under `build`) -> build_dataset.py CLI flag.
_BUILD_FLAGS = {
    "assembled_ratio": "--assembled-ratio",
    "seed": "--seed",
    "max_calls": "--max-calls",
    "max_shards": "--max-shards",
    "low_confidence_threshold": "--low-confidence-threshold",
}
# data-YAML key (under `build.prepare`) -> prepare_split.py CLI flag.
_PREPARE_FLAGS = {
    "wer_cap": "--wer-cap",
    "overlap_floor": "--overlap-floor",
    "agree_target_frac": "--agree-target-frac",
    "assembled_target_frac": "--assembled-target-frac",
    "entity_target_frac": "--entity-target-frac",
    "entity_min_name_word_len": "--entity-min-name-word-len",
    "entity_confidence_max": "--entity-confidence-max",
    "entity_confidence_max_freq": "--entity-confidence-max-freq",
    "entity_max_repeats": "--entity-max-repeats",
    "seed": "--seed",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a data YAML config (see configs/data/default.yaml).")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value",
        help="Override a single config value, e.g. --set build.assembled_ratio=0.3. Repeatable.",
    )
    p.add_argument("--raw-output", default=None,
                    help="Where build_dataset.py writes its raw output (source: build only). Defaults to the "
                         "config's build.raw_output, or ./asr_dataset.jsonl if that's unset too.")
    p.add_argument("--output", default=None,
                    help="Where prepare_split.py writes the curated output (source: build only). Defaults to "
                         "the config's build.output, or ./asr_dataset_curated.jsonl if that's unset too -- "
                         "Config.load_config()'s data_config resolution (see config.py) assumes this same "
                         "default for source: build, so a config that overrides build.output here should set "
                         "it in the YAML, not just via this flag, or the two will disagree about where the "
                         "curated dataset actually is.")
    return p.parse_args()


def _apply_overrides(cfg: dict, raw_overrides: list[str]) -> dict:
    for item in raw_overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects section.key=value, got: {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def _flags_from(section: dict, mapping: dict) -> list[str]:
    argv = []
    for key, flag in mapping.items():
        if key in section:
            argv += [flag, str(section[key])]
    return argv


def run_prebuilt(cfg: dict) -> None:
    dataset_id = (cfg.get("prebuilt") or {}).get("dataset_id")
    if not dataset_id:
        print("source: prebuilt but no prebuilt.dataset_id set in the config.", file=sys.stderr)
        sys.exit(1)
    try:
        info = HfApi().dataset_info(dataset_id)
    except Exception as e:
        print(f"Could not reach dataset repo {dataset_id}: {e}", file=sys.stderr)
        sys.exit(1)
    n_files = len(info.siblings) if info.siblings else 0
    print(f"Using prebuilt dataset {dataset_id} (last_modified={info.lastModified}, {n_files} files).")
    print(f"Set training.dataset_id: {dataset_id} in your train config -- no local build needed.")


_DEFAULT_RAW_OUTPUT = "./asr_dataset.jsonl"
_DEFAULT_OUTPUT = "./asr_dataset_curated.jsonl"


def run_build(cfg: dict, raw_output: str | None, curated_output: str | None) -> None:
    build_cfg = cfg.get("build") or {}
    raw_output = raw_output or build_cfg.get("raw_output", _DEFAULT_RAW_OUTPUT)
    curated_output = curated_output or build_cfg.get("output", _DEFAULT_OUTPUT)
    prepare_cfg = build_cfg.get("prepare") or {}
    if "assembled_ratio" not in build_cfg:
        print("build.assembled_ratio is required in the data config for source: build.", file=sys.stderr)
        sys.exit(1)

    build_argv = [sys.executable, str(REPO_ROOT / "src" / "build_dataset.py"),
                  *_flags_from(build_cfg, _BUILD_FLAGS), "--output-dir", raw_output]
    print("Running:", " ".join(build_argv), file=sys.stderr)
    subprocess.run(build_argv, check=True, cwd=REPO_ROOT)

    prepare_argv = [sys.executable, str(REPO_ROOT / "src" / "prepare_split.py"),
                     "--input", raw_output, "--output", curated_output,
                     *_flags_from(prepare_cfg, _PREPARE_FLAGS)]
    print("Running:", " ".join(prepare_argv), file=sys.stderr)
    subprocess.run(prepare_argv, check=True, cwd=REPO_ROOT)

    print(f"Curated dataset written to {curated_output}.")
    print(f"Set training.dataset_id: {curated_output} in your train config, "
          f"or upload it to a Hub dataset repo and point training.dataset_id there instead.")


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _apply_overrides(cfg, args.overrides)

    source = cfg.get("source")
    if source == "prebuilt":
        run_prebuilt(cfg)
    elif source == "build":
        run_build(cfg, args.raw_output, args.output)
    else:
        print(f"data config's 'source' must be 'prebuilt' or 'build', got {source!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
