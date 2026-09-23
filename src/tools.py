"""The bounded set of operations the Telegram agent can perform.

Every tool here is a plain, typed, docstringed Python function decorated
with `@beta_tool` -- the Anthropic SDK's tool runner derives each tool's
JSON schema from the signature + docstring, so there's no separate schema
file to keep in sync (see the `claude-api` skill's `tool-use.md`).

Deliberately NOT a general "run a shell command" tool -- every capability the
agent has is one of these named, scoped functions, so what it can do is
fixed and auditable rather than "whatever it decides to type." State-changing
tools take a `confirmed` parameter and gate their side effect on it (see
each one's docstring) rather than executing on the first ask.
"""
from __future__ import annotations

import glob
import json
import random
import subprocess
import sys
from pathlib import Path

from anthropic import beta_tool
from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervise  # noqa: E402
from config import load_config  # noqa: E402
from hub_sync import sync_output_dir  # noqa: E402
from notify import send_telegram_message, send_telegram_photo  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _confirm_gate(confirmed: bool, description: str) -> str | None:
    """Returns a message to show the user (without doing anything) if this
    action hasn't been confirmed yet; returns None if the caller should go
    ahead and actually perform it."""
    if confirmed:
        return None
    return (
        f"NOT YET EXECUTED (confirmed=False). Proposed action: {description}\n"
        "If the user confirms in their next message, call this exact same "
        "tool again with confirmed=True."
    )


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _deep_merge_inplace(base: dict, overrides: dict) -> dict:
    """Like _deep_merge, but mutates `base` directly instead of copying it at
    each level -- required for ruamel.yaml's CommentedMap, whose comments
    are tracked on the mapping object itself. `dict(base)` (what
    _deep_merge does) silently drops that tracking for every level it
    copies, even ones it doesn't otherwise touch (verified directly: a
    single-field edit_config() call deleted this project's config file's
    top-of-file comment block and every section-header comment on a
    modified section, while an untouched sibling section's comments
    survived untouched)."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_inplace(base[key], value)
        else:
            base[key] = value
    return base


def _make_ryaml() -> YAML:
    """A ruamel.yaml round-trip loader/dumper configured to match this
    project's YAML style: explicit "null" (ruamel's default representer
    renders a None value as blank instead, e.g. "eval_split:" instead of
    "eval_split: null" -- semantically identical but a needless style
    diff on every unrelated null-valued key in the file for a one-field
    edit)."""
    ryaml = YAML()
    ryaml.preserve_quotes = True

    def _represent_none(representer, data):
        return representer.represent_scalar("tag:yaml.org,2002:null", "null")

    ryaml.representer.add_representer(type(None), _represent_none)
    return ryaml


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------

@beta_tool
def check_training_status() -> str:
    """Reports whether a supervised training run is currently active, and if
    so, which config/output_dir it's using and how many auto-recovery
    retries have happened so far.
    """
    state = supervise.read_state()
    if not state:
        return "No training run has been started by this agent."
    alive = supervise.is_process_alive(state.get("supervisor_pid", -1))
    return (
        f"label={state.get('label')} status={state.get('status')} "
        f"(supervisor process alive={alive}) config={state.get('config_path')} "
        f"output_dir={state.get('output_dir')} attempt={state.get('attempt')} "
        f"started_at={state.get('started_at')}"
    )


@beta_tool
def stop_training(confirmed: bool = False) -> str:
    """Gracefully stops the currently active supervised training run (SIGTERM,
    not a hard kill -- lets an in-progress checkpoint save finish).

    Args:
        confirmed: Only pass True if the user has already explicitly
            confirmed this exact stop request in a prior message. Defaults
            to False, which reports what would happen without stopping
            anything.
    """
    state = supervise.read_state()
    if not state or state.get("status") != "running":
        return "No active training run to stop."
    pending = _confirm_gate(confirmed, f"stop the training run '{state.get('label')}' (output_dir={state.get('output_dir')})")
    if pending:
        return pending
    result = supervise.request_stop()
    return json.dumps(result)


@beta_tool
def run_finetune(config_path: str, overrides: dict | None = None, confirmed: bool = False) -> str:
    """Launches a new supervised fine-tuning run in the background (returns
    immediately -- training takes hours, so this does not block). The run
    gets automatic CUDA-OOM recovery (reduced batch size, resumed from the
    last checkpoint) -- see the training supervisor docs.

    Args:
        config_path: Path to a YAML config, e.g. "configs/qwen3.5-2b.yaml".
        overrides: Optional nested dict of config overrides, e.g.
            {"training": {"num_train_epochs": 1.0}}. Leave empty to just use
            the config file as-is.
        confirmed: Only pass True if the user has already explicitly
            confirmed starting this exact run in a prior message.
    """
    existing = supervise.read_state()
    if existing and existing.get("status") == "running" and supervise.is_process_alive(existing.get("supervisor_pid", -1)):
        return f"Refused: a run is already active ({existing.get('label')}). Stop it first."
    pending = _confirm_gate(confirmed, f"start training with config={config_path}, overrides={overrides or {}}")
    if pending:
        return pending
    result = supervise.launch_supervised_run(config_path, overrides or {})
    return json.dumps(result)


@beta_tool
def resume_training(config_path: str, checkpoint: str, overrides: dict | None = None, confirmed: bool = False) -> str:
    """Resumes training from a specific checkpoint path (distinct from
    run_finetune's fresh start). Also launched under the auto-recovery
    supervisor, in the background.

    Args:
        config_path: Path to a YAML config.
        checkpoint: Local checkpoint directory to resume from, e.g.
            "outputs/qwen3.5-2b/checkpoint-4000" or
            "outputs/qwen3.5-2b/best_checkpoint_wer".
        overrides: Optional additional config overrides.
        confirmed: Only pass True once the user has explicitly confirmed.
    """
    existing = supervise.read_state()
    if existing and existing.get("status") == "running" and supervise.is_process_alive(existing.get("supervisor_pid", -1)):
        return f"Refused: a run is already active ({existing.get('label')}). Stop it first."
    merged = _deep_merge(overrides or {}, {"resume_from_checkpoint": checkpoint})
    pending = _confirm_gate(confirmed, f"resume training with config={config_path} from checkpoint={checkpoint}")
    if pending:
        return pending
    result = supervise.launch_supervised_run(config_path, merged)
    return json.dumps(result)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@beta_tool
def edit_config(config_path: str, updates: dict, confirmed: bool = False) -> str:
    """Applies field updates to a YAML config file on disk, validating the
    result before writing (the file is left untouched if validation fails).

    Args:
        config_path: Path to the YAML config to edit.
        updates: Nested dict matching the config's own section structure,
            e.g. {"training": {"per_device_train_batch_size": 1}} or a
            top-level key like {"learning_rate": 0.0001}.
        confirmed: Only pass True once the user has explicitly confirmed
            this exact change.
    """
    pending = _confirm_gate(confirmed, f"edit {config_path} with updates={updates}")
    if pending:
        return pending

    path = Path(config_path)
    ryaml = _make_ryaml()
    with path.open() as f:
        raw = ryaml.load(f) or {}
    merged = _deep_merge_inplace(raw, updates)

    import io
    import tempfile
    buf = io.StringIO()
    ryaml.dump(merged, buf)
    new_text = buf.getvalue()

    try:
        # Validate against a temp copy, not the real path, so a bad edit
        # never touches the checked-in file.
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(new_text)
            tmp_path = f.name
        load_config(tmp_path)
    except Exception as e:
        return f"Validation failed, {config_path} was NOT modified: {e}"

    path.write_text(new_text)
    return f"Updated {config_path}: {updates}"


@beta_tool
def get_effective_config(output_dir: str) -> str:
    """Reads back the fully-resolved config a run actually used (train.py
    writes <output_dir>/config.yaml at startup, capturing every --set
    override applied on top of the YAML file) -- useful since overrides
    aren't otherwise visible without digging into this file.

    Args:
        output_dir: The run's output directory.
    """
    path = Path(output_dir) / "config.yaml"
    if not path.exists():
        return f"No config.yaml found under {output_dir}"
    return path.read_text()


@beta_tool
def validate_config(config_path: str) -> str:
    """Dry-runs config loading/validation without launching anything --
    catches typo'd keys or bad values before a real run.

    Args:
        config_path: Path to the YAML config to validate.
    """
    try:
        cfg = load_config(config_path)
        return f"Valid. model_id={cfg.model_id}, dataset_id={cfg.dataset_id}, output_dir={cfg.output_dir}"
    except Exception as e:
        return f"Invalid: {e}"


@beta_tool
def list_configs() -> str:
    """Lists every YAML config file under configs/, so you can see what's
    available before inspecting, copying, or launching one."""
    paths = sorted(glob.glob(str(REPO_ROOT / "configs" / "*.yaml")))
    if not paths:
        return "No config files found under configs/"
    return "\n".join(str(Path(p).relative_to(REPO_ROOT)) for p in paths)


@beta_tool
def get_config(config_path: str) -> str:
    """Returns a config file's full raw contents (comments included), e.g.
    to review one before editing or launching it. Distinct from
    get_effective_config, which reads back a *run's* already-resolved
    config.yaml, not a source configs/*.yaml file that hasn't been run yet.

    Args:
        config_path: Path to the YAML config, e.g. "configs/qwen3.5-2b.yaml".
    """
    path = Path(config_path)
    if not path.exists():
        return f"No config file found at {config_path}"
    return path.read_text()


@beta_tool
def copy_config(source_config_path: str, new_config_path: str, updates: dict | None = None, confirmed: bool = False) -> str:
    """Copies an existing config to a new file, optionally applying field
    updates on the copy -- e.g. to fork a config onto a new Hub repo/output
    directory for a new run without touching the original or its
    checkpoints (a different hub.repo_id/hub.folder means the new run's
    syncs land somewhere else entirely, so the old run's Hub folder is
    never touched). Comments are preserved (ruamel.yaml round-trip), same
    as edit_config. Refuses if new_config_path already exists -- use
    edit_config on it instead.

    Args:
        source_config_path: Existing config to copy from.
        new_config_path: Where to write the new config (must not already exist).
        updates: Optional field updates to apply to the copy, e.g.
            {"hub": {"repo_id": "PedramR/new-repo", "folder": null},
            "output_dir": "./outputs/new-run"}.
        confirmed: Only pass True once the user has explicitly confirmed.
    """
    source = Path(source_config_path)
    dest = Path(new_config_path)
    if not source.exists():
        return f"Source config not found: {source_config_path}"
    if dest.exists():
        return f"Refused: {new_config_path} already exists -- use edit_config to modify it instead."

    pending = _confirm_gate(confirmed, f"copy {source_config_path} to {new_config_path} with updates={updates or {}}")
    if pending:
        return pending

    ryaml = _make_ryaml()
    with source.open() as f:
        data = ryaml.load(f) or {}
    if updates:
        data = _deep_merge_inplace(data, updates)

    import io
    import tempfile
    buf = io.StringIO()
    ryaml.dump(data, buf)
    new_text = buf.getvalue()

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(new_text)
            tmp_path = f.name
        load_config(tmp_path)
    except Exception as e:
        return f"Validation failed, {new_config_path} was NOT created: {e}"

    dest.write_text(new_text)
    return f"Created {new_config_path} (copied from {source_config_path}, updates={updates or {}})"


# --------------------------------------------------------------------------
# Data & checkpoints
# --------------------------------------------------------------------------

@beta_tool
def build_data(assembled_ratio: float, output_name: str, max_calls: int | None = None, confirmed: bool = False) -> str:
    """Builds and curates a fresh training dataset in the background (can
    take tens of minutes for a full build -- returns immediately, sends a
    Telegram message when it finishes).

    Args:
        assembled_ratio: Fraction (0-1) of calls emitted as full-channel
            "assembled" rows rather than per-segment "chunked" rows.
        output_name: Base name for the output files, e.g. "asr_dataset_v2"
            -> writes ./asr_dataset_v2.jsonl and ./asr_dataset_v2_curated.jsonl.
        max_calls: Optional cap on calls processed, for a quick smaller build.
        confirmed: Only pass True once the user has explicitly confirmed.
    """
    pending = _confirm_gate(
        confirmed,
        f"build a new dataset (assembled_ratio={assembled_ratio}, max_calls={max_calls}, output_name={output_name})",
    )
    if pending:
        return pending

    raw_path = REPO_ROOT / f"{output_name}.jsonl"
    curated_path = REPO_ROOT / f"{output_name}_curated.jsonl"
    log_path = REPO_ROOT / f"{output_name}_build.log"

    build_cmd = [sys.executable, str(REPO_ROOT / "src" / "build_dataset.py"),
                 "--assembled-ratio", str(assembled_ratio), "--output-dir", str(raw_path)]
    if max_calls is not None:
        build_cmd += ["--max-calls", str(max_calls)]
    curate_cmd = [sys.executable, str(REPO_ROOT / "src" / "prepare_split.py"),
                  "--input", str(raw_path), "--output", str(curated_path)]

    script = (
        f"import subprocess, sys\n"
        f"from notify import send_telegram_message\n"
        f"r1 = subprocess.run({build_cmd!r})\n"
        f"if r1.returncode != 0:\n"
        f"    send_telegram_message('build_data: build_dataset.py failed, exit ' + str(r1.returncode))\n"
        f"    sys.exit(1)\n"
        f"r2 = subprocess.run({curate_cmd!r})\n"
        f"msg = 'build_data finished: {raw_path} / {curated_path}' if r2.returncode == 0 else "
        f"'build_data: prepare_split.py failed, exit ' + str(r2.returncode)\n"
        f"send_telegram_message(msg)\n"
    )
    with open(log_path, "w") as logf:
        subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
        )
    return f"Started in the background. Log: {log_path}. You'll get a Telegram message when it finishes."


@beta_tool
def list_checkpoints(output_dir: str) -> str:
    """Lists saved checkpoint directories for a run, with their step number
    and last-modified time.

    Args:
        output_dir: The run's output directory.
    """
    paths = sorted(glob.glob(str(Path(output_dir) / "checkpoint-*")))
    if not paths:
        return f"No checkpoint-* directories found under {output_dir}"
    lines = []
    for p in paths:
        step = Path(p).name.split("-")[-1]
        mtime = Path(p).stat().st_mtime
        lines.append(f"step {step}: {p} (mtime={mtime})")
    return "\n".join(lines)


@beta_tool
def check_hf_upload(repo_id: str, repo_type: str = "model") -> str:
    """Checks a Hugging Face Hub repo's contents and last-modified time.

    Args:
        repo_id: e.g. "PedramR/ASR_Post-processing" or
            "PedramR/ASR_Post-processing-dataset".
        repo_type: "model" or "dataset".
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().repo_info(repo_id, repo_type=repo_type)
    except Exception as e:
        return f"Could not fetch {repo_type} repo {repo_id}: {e}"
    n_files = len(info.siblings) if info.siblings else 0
    return f"{repo_id} ({repo_type}): last_modified={info.lastModified}, {n_files} files, private={info.private}"


@beta_tool
def sync_to_hub(output_dir: str, confirmed: bool = False) -> str:
    """Force-pushes a run's output_dir to the Hub right now, instead of
    waiting for the next checkpoint save to trigger it automatically.

    Args:
        output_dir: The run's output directory (must have a config.yaml,
            i.e. must have been produced by train.py).
        confirmed: Only pass True once the user has explicitly confirmed.
    """
    pending = _confirm_gate(confirmed, f"sync {output_dir} to the Hub now")
    if pending:
        return pending
    cfg = load_config(str(Path(output_dir) / "config.yaml"))
    sync_output_dir(cfg, commit_message="manual sync via agent")
    return f"Synced {output_dir} to {cfg.hub_repo_id}/{cfg.hub_repo_folder or Path(output_dir).name}"


# --------------------------------------------------------------------------
# Results & metrics
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@beta_tool
def get_checkpoint_metrics(output_dir: str, which: str = "latest") -> str:
    """Reads test-set metrics for a run.

    Args:
        output_dir: The run's output directory.
        which: "latest" (the most recent checkpoint eval) or "best" (the
            best-WER checkpoint tracked so far).
    """
    if which == "best":
        data = _read_json(Path(output_dir) / "best_checkpoint_wer" / "best_metrics.json")
    elif which == "latest":
        data = _read_json(Path(output_dir) / "test_eval" / "metrics.json")
    else:
        return "which must be 'latest' or 'best'"
    return json.dumps(data) if data else f"No {which} metrics found under {output_dir}"


@beta_tool
def get_secondary_eval_metrics(output_dir: str, which: str) -> str:
    """Reads the named-entity or typo/dictation-form eval-slice metrics,
    tracked separately from the aggregate test WER.

    Args:
        output_dir: The run's output directory.
        which: "entity" or "typo".
    """
    if which not in ("entity", "typo"):
        return "which must be 'entity' or 'typo'"
    data = _read_json(Path(output_dir) / f"test_eval_{which}" / "metrics.json")
    return json.dumps(data) if data else f"No {which} eval metrics found under {output_dir}"


def _predictions_path(output_dir: str, which: str) -> Path:
    subdir = {"main": "test_eval", "entity": "test_eval_entity", "typo": "test_eval_typo"}.get(which, "test_eval")
    return Path(output_dir) / subdir / "predictions.jsonl"


@beta_tool
def sample_predictions(output_dir: str, which: str = "main", n: int = 5) -> str:
    """Returns a random sample of predictions (input/output/reference/wer/...)
    from a run's test-set eval.

    Args:
        output_dir: The run's output directory.
        which: "main" (full test set), "entity", or "typo" eval slice.
        n: How many rows to sample.
    """
    path = _predictions_path(output_dir, which)
    if not path.exists():
        return f"No predictions found at {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sample = random.sample(rows, min(n, len(rows)))
    return json.dumps(sample, ensure_ascii=False, indent=2)


_ALLOWED_FIELDS = {"wer", "wer_zwnj_normalized", "input_overlap_pct", "hallucinated", "input", "output", "reference"}
_ALLOWED_OPS = {"==", "!=", ">", ">=", "<", "<=", "contains"}


def apply_prediction_filter(row: dict, filt: dict) -> bool:
    """Pure function: evaluates one filter {"field", "op", "value"} against
    one prediction row. Deliberately a tiny fixed-vocabulary DSL -- not an
    arbitrary expression/code string -- so this can't become a general code
    execution path.
    """
    field, op, value = filt.get("field"), filt.get("op"), filt.get("value")
    if field not in _ALLOWED_FIELDS:
        raise ValueError(f"unsupported field {field!r}, must be one of {sorted(_ALLOWED_FIELDS)}")
    if op not in _ALLOWED_OPS:
        raise ValueError(f"unsupported op {op!r}, must be one of {sorted(_ALLOWED_OPS)}")
    actual = row.get(field)
    if op == "contains":
        return isinstance(actual, str) and str(value) in actual
    if actual is None:
        return False
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    if op == ">":
        return actual > value
    if op == ">=":
        return actual >= value
    if op == "<":
        return actual < value
    if op == "<=":
        return actual <= value
    raise AssertionError("unreachable")


@beta_tool
def query_predictions(output_dir: str, filter: dict, which: str = "main", max_results: int = 10) -> str:
    """Filters a run's predictions by a simple field/operator/value
    condition -- e.g. find the highest-WER rows, or every hallucinated
    prediction.

    Args:
        output_dir: The run's output directory.
        filter: {"field": one of wer/wer_zwnj_normalized/input_overlap_pct/
            hallucinated/input/output/reference, "op": one of
            ==/!=/>/>=/</<=/contains, "value": the value to compare against}.
        which: "main", "entity", or "typo" eval slice.
        max_results: Cap on how many matching rows to return.
    """
    path = _predictions_path(output_dir, which)
    if not path.exists():
        return f"No predictions found at {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    try:
        matches = [r for r in rows if apply_prediction_filter(r, filter)]
    except ValueError as e:
        return f"Bad filter: {e}"
    return json.dumps(matches[:max_results], ensure_ascii=False, indent=2)


@beta_tool
def compare_runs(output_dir_a: str, output_dir_b: str) -> str:
    """Returns both runs' latest and best-so-far metrics, for the caller to
    compare (e.g. qwen vs. gemma, or current vs. best checkpoint).

    Args:
        output_dir_a: First run's output directory.
        output_dir_b: Second run's output directory.
    """
    def snapshot(d):
        return {
            "latest": _read_json(Path(d) / "test_eval" / "metrics.json"),
            "best": _read_json(Path(d) / "best_checkpoint_wer" / "best_metrics.json"),
        }
    return json.dumps({output_dir_a: snapshot(output_dir_a), output_dir_b: snapshot(output_dir_b)}, indent=2)


def _tb_dir_for(output_dir: str) -> Path:
    """The run's actual TensorBoard directory -- reads it back from the
    run's own config.yaml (tensorboard.logging_dir) rather than assuming
    the default, since that field can be overridden per-config; falls back
    to <output_dir>/tb (train.py's own default) if config.yaml is missing
    or doesn't set it.
    """
    config_path = Path(output_dir) / "config.yaml"
    if config_path.exists():
        try:
            cfg = load_config(config_path)
            if cfg.tensorboard_logging_dir:
                return Path(cfg.tensorboard_logging_dir)
        except Exception:
            pass
    return Path(output_dir) / "tb"


# These are namespaced with "/", not "_" -- HF Trainer's TensorBoardCallback
# runs every logged dict through rewrite_logs() before writing scalars,
# which renames any "eval_x" key to "eval/x", "test_x" to "test/x", and
# everything else (including Trainer's own internal "loss" from the
# training loop) to "train/x" -- verified directly against the installed
# transformers version's actual rewrite_logs(), not assumed. So "test_wer"
# (what train.py/evaluate.py actually call trainer.log() with) really lands
# in TensorBoard as "test/wer", and Trainer's internal per-step training
# loss (key "loss") lands as "train/loss", never "train_loss".
_DEFAULT_TAG_PRIORITY = ["test/wer", "test/best_wer", "test/entity_wer", "test/typo_wer", "eval/loss", "train/loss"]


def _tb_available_tags(tb_dir: Path) -> list[str]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    return sorted(ea.Tags().get("scalars", []))


def _read_tb_scalars(tb_dir: Path, tag: str) -> list[tuple[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})  # 0 -> load every point, not a sample
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


@beta_tool
def list_available_metrics(output_dir: str) -> str:
    """Lists every TensorBoard scalar metric tag available for a run --
    call this first to see what's plottable, then pass the ones you want
    to plot_metrics's `tags` argument.

    Args:
        output_dir: The run's output directory.
    """
    tb_dir = _tb_dir_for(output_dir)
    if not tb_dir.exists():
        return f"No TensorBoard directory found at {tb_dir}"
    tags = _tb_available_tags(tb_dir)
    if not tags:
        return f"No scalar metrics found under {tb_dir}"
    return json.dumps(tags)


@beta_tool
def plot_metrics(output_dir: str, tags: list[str] | None = None, title: str | None = None) -> str:
    """Renders a chart of one or more TensorBoard scalar metrics over
    training steps and sends it directly to Telegram as an image (not
    through this tool's own text return, since there's no reason for the
    model itself to see pixel data -- only the user needs the picture).

    Args:
        output_dir: The run's output directory (TensorBoard logs are read
            from wherever that run's own config.yaml points
            tensorboard.logging_dir, defaulting to <output_dir>/tb).
        tags: Which scalar tags to plot together on one chart, e.g.
            ["test/wer", "test/entity_wer"] or ["train/loss", "eval/loss"]
            (note the "/" -- HF Trainer namespaces logged metrics this way,
            not with "_"; call list_available_metrics first if unsure of
            the exact tag name). Leave empty to auto-pick from whatever of
            test/wer, test/best_wer, test/entity_wer, test/typo_wer,
            eval/loss, train/loss is actually present.
        title: Optional chart title. Defaults to output_dir's basename.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless -- no display on the training box
    import matplotlib.pyplot as plt

    tb_dir = _tb_dir_for(output_dir)
    if not tb_dir.exists():
        return f"No TensorBoard directory found at {tb_dir}"

    candidate_tags = tags or _DEFAULT_TAG_PRIORITY
    series_by_tag = {}
    for tag in candidate_tags:
        points = _read_tb_scalars(tb_dir, tag)
        if points:
            series_by_tag[tag] = points
        if not tags and series_by_tag:
            # Auto-pick mode: stop at the first priority tag(s) that
            # actually has data rather than dumping every metric on one
            # chart -- explicit `tags` from the caller are all plotted
            # regardless, since that's a deliberate comparison request.
            break

    if not series_by_tag:
        available = _tb_available_tags(tb_dir)
        return f"No matching scalar data under {tb_dir} for tags={candidate_tags}. Available scalar tags: {available}"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag, points in series_by_tag.items():
        steps, values = zip(*points)
        ax.plot(steps, values, marker=".", label=tag)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(title or Path(output_dir).name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(output_dir) / "agent_chart.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    caption = f"{Path(output_dir).name}: " + ", ".join(series_by_tag.keys())
    sent = send_telegram_photo(str(out_path), caption=caption)
    summary = {tag: {"n_points": len(pts), "latest": pts[-1][1], "min": min(v for _, v in pts), "max": max(v for _, v in pts)}
               for tag, pts in series_by_tag.items()}
    return f"{'Sent' if sent else 'Rendered (Telegram send failed, see logs)'} chart for {list(series_by_tag)}. Summary: {json.dumps(summary)}"


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

@beta_tool
def run_sweep(sweep_config: str, confirmed: bool = False) -> str:
    """Launches a sweep (multiple training configs run sequentially, then
    ranked by test WER) in the background. After ranking, the winner is
    automatically continued with a longer follow-up run -- see run_sweep.py.

    Args:
        sweep_config: Path to a sweep YAML, e.g. "configs/sweep.yaml".
        confirmed: Only pass True once the user has explicitly confirmed.
    """
    pending = _confirm_gate(confirmed, f"start a sweep from {sweep_config}")
    if pending:
        return pending
    log_path = REPO_ROOT / "sweep_agent.log"
    with open(log_path, "w") as logf:
        subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "src" / "run_sweep.py"), "--sweep", sweep_config],
            cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
        )
    return f"Sweep started in the background. Log: {log_path}."


# --------------------------------------------------------------------------
# Operational health
# --------------------------------------------------------------------------

@beta_tool
def check_gpu() -> str:
    """Runs nvidia-smi and returns its output."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        return result.stdout or result.stderr
    except FileNotFoundError:
        return "nvidia-smi not found on this machine (no NVIDIA GPU/driver here)."


@beta_tool
def check_disk_usage() -> str:
    """Reports disk usage for the repo root and its outputs/data directories."""
    lines = []
    df = subprocess.run(["df", "-h", str(REPO_ROOT)], capture_output=True, text=True)
    lines.append(df.stdout)
    for sub in ("outputs", "data"):
        p = REPO_ROOT / sub
        if p.exists():
            du = subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True)
            lines.append(du.stdout.strip())
    return "\n".join(lines)


@beta_tool
def tail_log(n: int = 50) -> str:
    """Returns the last N lines of the current (or most recent) training
    run's log file.

    Args:
        n: Number of lines from the end of the log to return.
    """
    state = supervise.read_state()
    if not state or not state.get("log_path"):
        return "No training log to show -- no run has been started by this agent."
    log_path = Path(state["log_path"])
    if not log_path.exists():
        return f"Log file not found: {log_path}"
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


ALL_TOOLS = [
    check_training_status, stop_training, run_finetune, resume_training,
    edit_config, get_effective_config, validate_config,
    list_configs, get_config, copy_config,
    build_data, list_checkpoints, check_hf_upload, sync_to_hub,
    get_checkpoint_metrics, get_secondary_eval_metrics, sample_predictions,
    query_predictions, compare_runs, plot_metrics, list_available_metrics,
    run_sweep,
    check_gpu, check_disk_usage, tail_log,
]
