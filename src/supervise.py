"""Runs one training config as a supervised, auto-recovering subprocess.

Meant to be launched as its own **detached background process** (see
`launch_supervised_run()`) -- not imported and called for a blocking,
in-process training run. This keeps the caller (the Telegram agent's event
loop) responsive while training runs for hours: the agent just starts this
as a detached child and returns immediately. This process itself launches
and monitors `train.py`, retries on a detected CUDA OOM, and sends Telegram
notifications independently of whether the process that started it is even
still running.

State (pid, output_dir, config_path, status, ...) is persisted to a JSON
file so any other process (a restarted agent, a `stop` request) can
discover/control the run without holding an in-memory handle across process
boundaries.

Detection/recovery is deliberately narrow: only a recognized CUDA-OOM
signature triggers an automatic retry (with a reduced batch size); anything
else -- including a silent host-level (CPU RAM) OOM, which the Linux kernel
SIGKILLs with no traceback at all -- stops and alerts instead of guessing at
a fix.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config  # noqa: E402
from notify import send_telegram_message  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / ".agent_training_state.json"
TRAIN_SCRIPT = str(Path(__file__).resolve().parent / "train.py")

_OOM_SIGNATURES = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUDA error: out of memory",
)
MAX_RETRIES = 5
POLL_INTERVAL_SECS = 15
POST_STOP_SETTLE_SECS = 5  # let CUDA driver-side cleanup finish before relaunching


# --------------------------------------------------------------------------
# State file
# --------------------------------------------------------------------------

def read_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return None


def _write_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE_PATH)


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# --------------------------------------------------------------------------
# Failure classification and recovery math
# --------------------------------------------------------------------------

def _tail(path: Path, n_bytes: int = 8000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def classify_failure(log_tail: str, returncode: int) -> str:
    """Returns "oom", "host_oom", or "unknown" -- never guesses beyond this.

    A host-level (CPU RAM, not GPU VRAM) OOM gets the process SIGKILLed by
    the Linux kernel with no traceback at all (returncode -9) -- that's
    classified separately from "oom" (a real CUDA-OOM signature in the log)
    since it's not recoverable the same way and there's nothing to grep for.
    Both "host_oom" and "unknown" stop without retrying.
    """
    if returncode == -9:
        return "host_oom"
    if any(sig in log_tail for sig in _OOM_SIGNATURES):
        return "oom"
    return "unknown"


def compute_oom_recovery_overrides(config_path: str, prior_overrides: dict) -> dict | None:
    """Given the config as currently resolved (base file + prior_overrides),
    returns a new overrides dict with per_device_train_batch_size halved
    (floor 1), gradient_accumulation_steps doubled to preserve the effective
    batch size, test.batch_size halved (floor 1, a secondary precaution --
    the one OOM already documented in this repo happened during
    checkpoint-eval generation, not a training step), and
    resume_from_checkpoint set. Returns None if per_device_train_batch_size
    is already 1 -- there's nothing left to reduce.
    """
    cfg = load_config(config_path, overrides=prior_overrides)
    if cfg.per_device_train_batch_size <= 1:
        return None

    new_bs = max(1, cfg.per_device_train_batch_size // 2)
    scale = cfg.per_device_train_batch_size // new_bs
    new_overrides = {
        "training": {
            "per_device_train_batch_size": new_bs,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps * scale,
        },
        "test": {
            "batch_size": max(1, cfg.test_batch_size // 2),
        },
        "resume_from_checkpoint": True,
    }
    merged = _deep_merge(prior_overrides, new_overrides)
    return merged


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _overrides_to_set_args(overrides: dict, prefix: str = "") -> list[str]:
    """Flattens a nested overrides dict into repeated --set section.key=value
    CLI args, matching train.py's _parse_overrides()'s expected shape."""
    args = []
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(_overrides_to_set_args(value, prefix=full_key))
        else:
            args.append("--set")
            args.append(f"{full_key}={value}")
    return args


# --------------------------------------------------------------------------
# The supervised run itself (runs in the detached daemon process)
# --------------------------------------------------------------------------

_current_child: subprocess.Popen | None = None
_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    if _current_child is not None and _current_child.poll() is None:
        _current_child.send_signal(signal.SIGTERM)


def run_supervised(config_path: str, initial_overrides: dict, label: str | None) -> None:
    """Blocking. Meant to run as the entire lifetime of the detached daemon
    process -- see module docstring."""
    global _current_child, _stop_requested
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg = load_config(config_path, overrides=initial_overrides)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "supervisor_train.log"

    overrides = dict(initial_overrides)
    attempt = 0
    label = label or Path(config_path).stem

    send_telegram_message(f"[{label}] Starting supervised training run (config={config_path}).")

    while True:
        set_args = _overrides_to_set_args(overrides)
        cmd = [sys.executable, "-u", TRAIN_SCRIPT, "--config", config_path, *set_args]

        _write_state({
            "label": label,
            "config_path": config_path,
            "output_dir": str(output_dir),
            "overrides": overrides,
            "log_path": str(log_path),
            "status": "running",
            "attempt": attempt,
            "supervisor_pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        with open(log_path, "a") as logf:
            logf.write(f"\n\n=== supervise.py: attempt {attempt}, overrides={overrides} ===\n")
            logf.flush()
            _current_child = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
            )

        last_metrics_check = 0.0
        while True:
            ret = _current_child.poll()
            if ret is not None:
                break
            if time.time() - last_metrics_check > POLL_INTERVAL_SECS:
                last_metrics_check = time.time()
                _maybe_send_checkpoint_digest(output_dir, label)
            time.sleep(2)

        _current_child.wait()  # fully reap, per the reviewed design -- not just poll()

        if _stop_requested:
            _write_state({**(read_state() or {}), "status": "stopped"})
            send_telegram_message(f"[{label}] Training stopped as requested.")
            return

        if ret == 0:
            _write_state({**(read_state() or {}), "status": "completed"})
            send_telegram_message(f"[{label}] Training completed successfully.")
            return

        log_tail = _tail(log_path)
        failure = classify_failure(log_tail, ret)

        if failure == "oom" and attempt < MAX_RETRIES:
            new_overrides = compute_oom_recovery_overrides(config_path, overrides)
            if new_overrides is not None:
                attempt += 1
                send_telegram_message(
                    f"[{label}] CUDA OOM detected (attempt {attempt}/{MAX_RETRIES}). "
                    f"Reducing batch size and resuming from the last checkpoint. "
                    f"New overrides: {new_overrides.get('training', {})}, "
                    f"test.batch_size={new_overrides.get('test', {}).get('batch_size')}."
                )
                overrides = new_overrides
                time.sleep(POST_STOP_SETTLE_SECS)
                continue

        _write_state({**(read_state() or {}), "status": "failed", "failure_type": failure})
        send_telegram_message(
            f"[{label}] Training stopped -- {failure} (exit code {ret}), "
            f"no further automatic retry. Log tail:\n{log_tail[-1500:]}"
        )
        return


def _maybe_send_checkpoint_digest(output_dir: Path, label: str) -> None:
    """Appends any new test_eval/metrics.json reading to
    <output_dir>/supervisor_wer_history.jsonl and sends a Telegram digest --
    tracks a real trend (improving/plateaued/regressed), unlike
    best_checkpoint_wer/best_metrics.json, which only ever updates on
    improvement and so can't show a plateau or a regression.
    """
    metrics_path = output_dir / "test_eval" / "metrics.json"
    if not metrics_path.exists():
        return
    history_path = output_dir / "supervisor_wer_history.jsonl"
    try:
        mtime = metrics_path.stat().st_mtime
    except OSError:
        return

    last_seen_path = output_dir / ".supervisor_last_digest_mtime"
    last_seen = float(last_seen_path.read_text()) if last_seen_path.exists() else 0.0
    if mtime <= last_seen:
        return

    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return  # race with the writer -- skip, retry next poll (see evaluate.py's atomic write)

    reading = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wer": metrics.get("wer"),
        "exact_match": metrics.get("exact_match"),
        "hallucination_rate": metrics.get("hallucination_rate"),
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(reading) + "\n")
    last_seen_path.write_text(str(mtime))

    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    trend = _classify_trend([h["wer"] for h in history if h["wer"] is not None])
    send_telegram_message(
        f"[{label}] Checkpoint digest: wer={reading['wer']}, "
        f"exact_match={reading['exact_match']}, "
        f"hallucination_rate={reading['hallucination_rate']} -- trend: {trend}"
    )


def _classify_trend(wer_history: list[float], window: int = 3) -> str:
    """Pure function: given consecutive WER readings (lower is better), a
    rough (not statistically rigorous) trend classification:
    - the latest reading ties or beats the best ever seen -> "improving"
    - the latest reading is at least as good as the start of the recent
      window (not getting worse lately, even if not a new best) -> "plateaued"
    - otherwise, the recent window is trending worse -> "regressed for now"
    """
    if len(wer_history) < 2:
        return "not enough data yet"
    latest = wer_history[-1]
    best_so_far = min(wer_history)
    checkpoints_since_best = 0
    for w in reversed(wer_history):
        if w <= best_so_far:
            break
        checkpoints_since_best += 1
    if latest <= best_so_far:
        return "improving (new best)"
    recent = wer_history[-window:]
    if latest <= recent[0]:
        return f"plateaued ({checkpoints_since_best} checkpoints since best)"
    return f"regressed for now ({checkpoints_since_best} checkpoints since best)"


# --------------------------------------------------------------------------
# Public API for tools.py -- launch/stop/status without blocking the caller
# --------------------------------------------------------------------------

def launch_supervised_run(config_path: str, overrides: dict | None = None, label: str | None = None) -> dict:
    """Starts run_supervised() as a **detached background process** and
    returns immediately. Refuses if a previous run's state file still shows
    a live process (single GPU -- only one run at a time).
    """
    existing = read_state()
    if existing and existing.get("status") == "running" and is_process_alive(existing.get("supervisor_pid", -1)):
        return {"started": False, "reason": f"a run is already active: {existing}"}

    overrides = overrides or {}
    args = [sys.executable, str(Path(__file__).resolve()), "--config", config_path]
    if label:
        args += ["--label", label]
    args += _overrides_to_set_args(overrides)

    supervisor_log = Path(config_path).with_suffix("")  # placeholder base; real log goes under output_dir
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    return {"started": True, "supervisor_pid": proc.pid, "state_path": str(STATE_PATH)}


def request_stop() -> dict:
    state = read_state()
    if not state or state.get("status") != "running":
        return {"stopped": False, "reason": "no active run"}
    pid = state.get("supervisor_pid")
    if not pid or not is_process_alive(pid):
        return {"stopped": False, "reason": "recorded run is not actually alive"}
    os.kill(pid, signal.SIGTERM)
    return {"stopped": True, "pid": pid}


# --------------------------------------------------------------------------
# CLI entrypoint -- this IS the detached daemon process launch_supervised_run() starts
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.key=value")
    args = p.parse_args()

    overrides: dict = {}
    for item in args.overrides:
        key, _, value = item.partition("=")
        if "." in key:
            section, sub_key = key.split(".", 1)
            overrides.setdefault(section, {})[sub_key] = value
        else:
            overrides[key] = value

    run_supervised(args.config, overrides, args.label)


if __name__ == "__main__":
    main()
