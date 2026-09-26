"""Run a queue of training jobs sequentially, then compare their results.

Each job in the sweep YAML (see configs/train/sweep.yaml) names its own config file
-- jobs don't have to share one base config, so a run that needs genuinely
different settings (not just a couple of overridden values) just points at
its own full YAML file. An optional `overrides` on top of that file still
lets you tweak a couple of values without duplicating the whole thing (e.g.
reusing configs/train/base.yaml for most jobs, only overriding model_id).

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notify import send_telegram_message  # noqa: E402
from supervise import launch_supervised_run  # noqa: E402


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


# The final eval's metrics.json per slice (written by evaluate.py's
# run_test_eval()) -- read back here rather than recomputed, same as
# _test_wer() above.
_EVAL_SUBDIRS = {"main": "test_eval", "entity": "test_eval_entity", "typo": "test_eval_typo"}


def _eval_metrics(output_dir: Path) -> dict[str, dict[str, Any]]:
    metrics = {}
    for slice_name, subdir in _EVAL_SUBDIRS.items():
        metrics_path = output_dir / subdir / "metrics.json"
        if metrics_path.exists():
            metrics[slice_name] = json.loads(metrics_path.read_text())
    return metrics


def _format_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v if v is not None else 'n/a'}"
        for k, v in metrics.items()
    )


def _print_eval_metrics(eval_metrics: dict[str, dict[str, Any]], indent: str = "    ") -> None:
    for slice_name, metrics in eval_metrics.items():
        print(f"{indent}{slice_name:6s}  {_format_metrics(metrics)}", flush=True)


def _launch_followup(output_root: Path, winner: dict) -> None:
    """Auto-continues the sweep's winner with a longer follow-up run,
    resumed from its best checkpoint -- routed through
    launch_supervised_run() so it gets the same CUDA-OOM auto-recovery as
    any other supervised run.

    Reads the winner's own resolved config from <output_root>/<name>/config.yaml
    (written by train.py itself once training starts) rather than the raw
    sweep.yaml job entry, which lacks that job's own `overrides` (e.g. a
    different model_id). Uses a NEW output_dir (<name>-followup) rather than
    reusing the winner's own directory, so the follow-up's artifacts don't
    overwrite the sweep entry's original results in place.
    """
    name = winner["name"]
    job_dir = output_root / name
    resolved_config_path = job_dir / "config.yaml"
    if not resolved_config_path.exists():
        send_telegram_message(f"Sweep finished; winner '{name}' has no config.yaml to build a follow-up from -- skipping.")
        return

    best_checkpoint = job_dir / "best_checkpoint_wer"
    if not best_checkpoint.exists():
        send_telegram_message(f"Sweep finished; winner '{name}' has no best_checkpoint_wer/ to resume from -- skipping follow-up.")
        return

    followup_dir = output_root / f"{name}-followup"
    overrides = {
        "output_dir": str(followup_dir),
        "resume_from_checkpoint": str(best_checkpoint.resolve()),
    }
    result = launch_supervised_run(str(resolved_config_path), overrides, label=f"{name}-followup")
    send_telegram_message(
        f"Sweep finished. Winner: '{name}' (test_wer={winner['test_wer']}). "
        f"Launching follow-up run at {followup_dir} resumed from its best checkpoint. {result}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="Path to a sweep YAML file (see configs/train/sweep.yaml).")
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
        eval_metrics = _eval_metrics(output_root / name)
        results.append({
            "name": name, "status": status, "best_eval_loss": best_loss, "test_wer": test_wer,
            "eval_metrics": eval_metrics,
        })
        print(f"=== Job '{name}': {status}, best eval_loss={best_loss}, test_wer={test_wer} ===", flush=True)
        _print_eval_metrics(eval_metrics)

    print("\n=== Sweep summary ===")
    # test_wer is the metric that actually answers "which model corrects
    # ASR output best" -- eval_loss is only a fallback sort key when it's
    # not available (e.g. test.dataset_id unset).
    for r in sorted(results, key=lambda r: (r["test_wer"] is None, r["test_wer"], r["best_eval_loss"] is None, r["best_eval_loss"])):
        loss_str = f"{r['best_eval_loss']:.4f}" if r["best_eval_loss"] is not None else "n/a"
        wer_str = f"{r['test_wer']:.4f}" if r["test_wer"] is not None else "n/a"
        print(f"{r['name']:30s}  {r['status']:20s}  test_wer={wer_str}  best_eval_loss={loss_str}")
        _print_eval_metrics(r["eval_metrics"])

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written to {summary_path}")

    # Auto-continue the winner with a longer follow-up run. Guard against an
    # all-failed sweep explicitly: sorted(...)[0] still returns a job even
    # when every test_wer is None (the sort key just pushes None last, it
    # doesn't exclude it), so this must check for a real winner first.
    winner = sorted(results, key=lambda r: (r["test_wer"] is None, r["test_wer"], r["best_eval_loss"] is None, r["best_eval_loss"]))[0]
    if winner["test_wer"] is None:
        send_telegram_message("Sweep finished with no successful jobs (no test_wer recorded) -- skipping follow-up.")
    else:
        _launch_followup(output_root, winner)


if __name__ == "__main__":
    main()
