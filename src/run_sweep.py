"""Run a queue of training jobs sequentially, then compare their results.

Each job in the sweep YAML (see configs/sweep.yaml) names its own config file
-- jobs don't have to share one base config, so a run that needs genuinely
different settings (not just a couple of overridden values) just points at
its own full YAML file. An optional `overrides` on top of that file still
lets you tweak a couple of values without duplicating the whole thing (e.g.
reusing configs/base.yaml for most jobs, only overriding model_id).

Jobs run one after another -- not in parallel -- since they share one GPU. A
failed job doesn't stop the queue: it's recorded and the remaining jobs still
run, so a bad config for one model doesn't cost you the comparison for the
others.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _materialize_job_config(job: dict, output_root: Path) -> Path:
    """Merge `job`'s overrides onto its own config file, then namespace per-job paths.

    The namespacing (output_dir, tensorboard dir, hub folder) must check
    whether *this job's own overrides* set that key -- not whether the merged
    dict already has a value -- since the job's config file already sets all
    of these and a plain `setdefault` on the merged dict would never fire,
    leaving every job in the sweep writing to (and overwriting) the same
    output_dir/hub folder.

    hub.repo_id is deliberately left untouched: every job in a sweep shares
    ONE Hub repo (set in each job's own config), each getting its own
    hub.folder -- unlike output_dir/tensorboard dir, there's no separate
    per-job repo to generate here.
    """
    name = job["name"]
    if "config" not in job:
        raise ValueError(f"job '{name}' is missing a 'config' path")
    job_dir = output_root / name
    job_dir.mkdir(parents=True, exist_ok=True)

    job_raw = yaml.safe_load(Path(job["config"]).read_text()) or {}
    job_overrides = job.get("overrides", {})
    merged = _deep_merge(job_raw, job_overrides)

    if "output_dir" not in job_overrides:
        merged["output_dir"] = str(job_dir)

    tb_overrides = job_overrides.get("tensorboard", {})
    if "logging_dir" not in tb_overrides:
        merged.setdefault("tensorboard", {})
        merged["tensorboard"]["logging_dir"] = str(job_dir / "tb")

    hub_overrides = job_overrides.get("hub", {})
    if "folder" not in hub_overrides:
        merged.setdefault("hub", {})
        merged["hub"]["folder"] = name

    config_path = job_dir / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(merged, sort_keys=False))
    return config_path


def _best_eval_loss(output_dir: Path) -> float | None:
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    losses = [e["eval_loss"] for e in state.get("log_history", []) if "eval_loss" in e]
    return min(losses) if losses else None


def _test_wer(output_dir: Path) -> float | None:
    metrics_path = output_dir / "test_eval" / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text()).get("wer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="Path to a sweep YAML file (see configs/sweep.yaml).")
    parser.add_argument("--train_script", default=str(Path(__file__).with_name("train.py")))
    args = parser.parse_args()

    sweep = yaml.safe_load(Path(args.sweep).read_text())
    output_root = Path(sweep.get("output_root", "./outputs/sweep"))

    results: list[dict[str, Any]] = []
    for job in sweep["jobs"]:
        name = job["name"]
        config_path = _materialize_job_config(job, output_root)
        print(f"\n=== Running job '{name}' ({config_path}) ===", flush=True)

        proc = subprocess.run([sys.executable, args.train_script, "--config", str(config_path)])
        status = "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
        best_loss = _best_eval_loss(output_root / name)
        test_wer = _test_wer(output_root / name)
        results.append({"name": name, "status": status, "best_eval_loss": best_loss, "test_wer": test_wer})
        print(f"=== Job '{name}': {status}, best eval_loss={best_loss}, test_wer={test_wer} ===", flush=True)

    print("\n=== Sweep summary ===")
    # test_wer is the metric that actually answers "which model corrects
    # ASR output best" -- eval_loss is only a fallback sort key when it's
    # not available (e.g. test.dataset_id unset).
    for r in sorted(results, key=lambda r: (r["test_wer"] is None, r["test_wer"], r["best_eval_loss"] is None, r["best_eval_loss"])):
        loss_str = f"{r['best_eval_loss']:.4f}" if r["best_eval_loss"] is not None else "n/a"
        wer_str = f"{r['test_wer']:.4f}" if r["test_wer"] is not None else "n/a"
        print(f"{r['name']:30s}  {r['status']:20s}  test_wer={wer_str}  best_eval_loss={loss_str}")

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
