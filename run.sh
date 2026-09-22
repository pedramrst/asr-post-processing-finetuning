#!/usr/bin/env bash
# Entry point for running this pipeline on a fresh Vast.ai (or any bare Ubuntu
# + CUDA) instance. Assumes this repo is ALREADY on the instance -- there's no
# git remote for it yet, so this script can't pull itself there; copy it up
# first (git clone once you have a remote, rsync, scp, or Vast's file upload),
# then run this from the repo root.
#
# Usage:
#   ./run.sh smoke   (default) tiny end-to-end check: small local data slice,
#                     1 epoch, low save_steps -- exercises train/save/hub-sync/
#                     test-eval/WER cheaply before you commit to a real run.
#   ./run.sh build    (Re)build + curate the full training dataset only.
#   ./run.sh train <config> [--set key=value ...]
#                     Train just one config, e.g. `./run.sh train qwen3.5-2b`
#                     or `./run.sh train gemma-3-1b-it` (bare names resolve to
#                     configs/<name>.yaml; a path to any .yaml file also
#                     works). Anything after <config> is forwarded to
#                     train.py, e.g. `./run.sh train qwen3.5-2b --set
#                     train_fraction=0.1 --set output_dir=./outputs/qwen-10pct`
#                     for a data-scaling ablation run. Assumes the full
#                     dataset was already built (run `build` first). Unlike
#                     `sweep`, output_dir/hub.folder come straight from that
#                     config file, not auto-namespaced.
#   ./run.sh sweep    Run the model comparison (configs/sweep.yaml). Assumes
#                     the full dataset was already built (run `build` first).
#   ./run.sh full     build + sweep, back to back. Multi-hour, real GPU cost --
#                     run `smoke` first if you haven't already.
#   ./run.sh agent    Start the Telegram + Claude tool-calling operations
#                     agent (see README's "Telegram agent" section). Runs
#                     indefinitely, long-polling Telegram -- needs
#                     ANTHROPIC_API_KEY/TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID in
#                     .env.
#
# Runs inside tmux automatically (session name "run") so a dropped SSH
# connection doesn't kill a long build/sweep -- reattach with `tmux attach -t run`.
set -euo pipefail

MODE="${1:-smoke}"
CONFIG_ARG="${2:-}"
# Production corrects one full call channel at a time, so assembled
# (full-channel) rows should dominate the raw data -- see build_dataset.py's
# docstring. prepare_split.py's --assembled-target-frac then balances the
# curated split from whatever this produces.
ASSEMBLED_RATIO="${ASSEMBLED_RATIO:-0.9}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

log() { printf '\n=== %s ===\n' "$1"; }

# --- Validate the mode before doing any expensive setup work ---------------
case "$MODE" in
  smoke|build|sweep|full|agent) ;;
  train)
    if [ -z "$CONFIG_ARG" ]; then
      echo "Usage: ./run.sh train <config-name-or-path>" >&2
      echo "Available configs:" >&2
      ls configs/*.yaml | sed 's/^/  /' >&2
      exit 1
    fi
    ;;
  *)
    echo "Unknown mode '$MODE'. Usage: ./run.sh [smoke|build|train <config>|sweep|full|agent]" >&2
    exit 1
    ;;
esac

# --- Re-launch inside tmux so this survives an SSH disconnect -------------
if [ -z "${TMUX:-}" ] && [ -z "${RUN_SH_NO_TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    if [ "$(id -u)" = "0" ]; then
      apt-get update -qq && apt-get install -y -qq tmux
    else
      sudo apt-get update -qq && sudo apt-get install -y -qq tmux
    fi
  fi
  log "Launching inside tmux session 'run' (reattach any time with: tmux attach -t run)"
  ARGS="$(printf '%q ' "$@")"
  tmux new-session -d -s run "cd '$REPO_DIR' && RUN_SH_NO_TMUX=1 ./run.sh $ARGS; exec bash"
  tmux attach -t run
  exit 0
fi

# --- Sanity checks ----------------------------------------------------------
log "GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "No GPU detected (nvidia-smi failed) -- this pipeline requires CUDA. Aborting." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Python + venv -----------------------------------------------------------
log "Python environment"
if [ -f /venv/main/bin/activate ]; then
  # Vast.ai PyTorch templates ship a preconfigured venv at /venv/main with
  # torch already built against that image's CUDA version. Reusing it avoids
  # a slow, redundant torch re-download and the risk of pip resolving a torch
  # build that doesn't match the image. requirements.txt leaves `torch`
  # unpinned specifically so installing into this env below won't touch it.
  echo "Found /venv/main -- reusing it instead of creating a new venv." >&2
  source /venv/main/bin/activate
else
  PYBIN="python3.12"
  if ! command -v "$PYBIN" >/dev/null 2>&1; then
    echo "python3.12 not found, trying to install it..." >&2
    if [ "$(id -u)" = "0" ]; then
      apt-get update -qq && apt-get install -y -qq python3.12 python3.12-venv || true
    else
      sudo apt-get update -qq && sudo apt-get install -y -qq python3.12 python3.12-venv || true
    fi
  fi
  if ! command -v "$PYBIN" >/dev/null 2>&1; then
    echo "python3.12 unavailable, falling back to python3 (untested combination -- watch for dependency issues)." >&2
    PYBIN="python3"
  fi

  if [ ! -d .venv ]; then
    "$PYBIN" -m venv .venv
  fi
  source .venv/bin/activate
fi
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Optional, best-effort: speeds up Qwen3.5's hybrid SSM/linear-attention
# layers, used in both training and generation (eval). Without them,
# transformers falls back to a correct but much slower reference PyTorch
# path -- not fatal, so unlike requirements.txt above this is allowed to
# fail without aborting the whole script: these are CUDA/torch-version-
# sensitive compiled extensions, not guaranteed to build on every image.
log "Installing optional attention kernels (causal-conv1d, flash-linear-attention)"
pip install -q causal-conv1d flash-linear-attention || echo "Optional kernel install failed -- continuing without it (correct, just slower)." >&2

# Mid-training test-set generation (TestEvalCallback) allocates/frees many
# differently-sized KV-cache tensors in the same process as the Trainer,
# which fragments PyTorch's CUDA caching allocator badly enough to OOM a
# later training step's backward pass even when nominal free memory looks
# sufficient (observed directly: a later step failed to allocate 7.18GiB
# with 7.5GiB "reserved but unallocated" sitting right there). This is
# PyTorch's own suggested mitigation -- letting allocator segments grow/
# shrink instead of being fixed-size -- as defense-in-depth alongside the
# explicit torch.cuda.empty_cache() in evaluate.py's run_test_eval().
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- HF auth ------------------------------------------------------------
log "Hugging Face auth"
if [ ! -f .env ]; then
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "No .env found; writing one from the HF_TOKEN already set in this shell's environment." >&2
    printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > .env
  else
    cat >&2 <<'MSG'
No .env and no HF_TOKEN environment variable set. Either:
  - copy .env.example to .env and fill in a token with write access to the
    hub.repo_id configured in your configs (see configs/base.yaml), or
  - set HF_TOKEN in this shell (e.g. via Vast's instance environment
    variables) before running this script.
Aborting.
MSG
    exit 1
  fi
fi

run_build() {
  log "Building dataset (--assembled-ratio $ASSEMBLED_RATIO)"
  python src/build_dataset.py --assembled-ratio "$ASSEMBLED_RATIO" --output-dir ./asr_dataset.jsonl
  log "Curating dataset"
  python src/prepare_split.py --input ./asr_dataset.jsonl --output ./asr_dataset_curated.jsonl
}

run_smoke() {
  log "Building smoke-test data slice (50 calls)"
  python src/build_dataset.py --assembled-ratio "$ASSEMBLED_RATIO" --max-calls 50 --output-dir ./asr_dataset_smoke.jsonl
  python src/prepare_split.py --input ./asr_dataset_smoke.jsonl --output ./asr_dataset_smoke_curated.jsonl
  log "Running smoke-test training (configs/smoke.yaml)"
  python src/train.py --config configs/smoke.yaml
  log "Smoke test complete -- check outputs/smoke-test/ and the 'smoke-test' folder in your Hub repo."
}

run_sweep() {
  if [ ! -f ./asr_dataset_curated.jsonl ]; then
    echo "./asr_dataset_curated.jsonl not found -- run './run.sh build' first." >&2
    exit 1
  fi
  log "Running model comparison sweep (configs/sweep.yaml)"
  python src/run_sweep.py --sweep configs/sweep.yaml
}

run_agent() {
  if [ -z "${OPENROUTER_API_KEY:-}" ] && ! grep -q '^OPENROUTER_API_KEY=.' .env 2>/dev/null; then
    echo "OPENROUTER_API_KEY not set (env or .env) -- the agent needs this to call Claude via OpenRouter. Aborting." >&2
    exit 1
  fi
  if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] && ! grep -q '^TELEGRAM_BOT_TOKEN=.' .env 2>/dev/null; then
    echo "TELEGRAM_BOT_TOKEN not set (env or .env). Aborting." >&2
    exit 1
  fi
  if [ -z "${TELEGRAM_CHAT_ID:-}" ] && ! grep -q '^TELEGRAM_CHAT_ID=.' .env 2>/dev/null; then
    echo "TELEGRAM_CHAT_ID not set (env or .env). Aborting." >&2
    exit 1
  fi
  log "Starting Telegram agent (long-running -- reattach to this tmux session to check on it)"
  python src/telegram_agent.py
}

run_train() {
  local cfg="$1" config_path
  shift
  # Accept a bare name (resolved against configs/, with or without .yaml) or
  # any path to a .yaml file, so both `./run.sh train qwen3.5-2b` and
  # `./run.sh train configs/qwen3.5-2b.yaml` work. Anything after the config
  # (e.g. `--set train_fraction=0.1 --set output_dir=...`) is forwarded
  # straight to train.py.
  if [ -f "$cfg" ]; then
    config_path="$cfg"
  elif [ -f "configs/$cfg.yaml" ]; then
    config_path="configs/$cfg.yaml"
  elif [ -f "configs/$cfg" ]; then
    config_path="configs/$cfg"
  else
    echo "Config not found: '$cfg'. Available configs:" >&2
    ls configs/*.yaml | sed 's/^/  /' >&2
    exit 1
  fi
  if [ ! -f ./asr_dataset_curated.jsonl ]; then
    echo "./asr_dataset_curated.jsonl not found -- run './run.sh build' first." >&2
    exit 1
  fi
  log "Training $config_path"
  python src/train.py --config "$config_path" "$@"
}

case "$MODE" in
  smoke) run_smoke ;;
  build) run_build ;;
  train) run_train "$CONFIG_ARG" "${@:3}" ;;
  sweep) run_sweep ;;
  full)  run_build; run_sweep ;;
  agent) run_agent ;;
  *)
    echo "Unknown mode '$MODE'. Usage: ./run.sh [smoke|build|train <config>|sweep|full|agent]" >&2
    exit 1
    ;;
esac

log "Done ($MODE)"
