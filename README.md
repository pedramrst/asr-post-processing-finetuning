# callcc-2k ASR correction fine-tuning

LoRA SFT that teaches a small LLM to correct our in-house Whisper ASR output
(`text_whisper`) into the frontier-model transcript (`text`), using Persian
call-center audio metadata from `ErfanRou/callcc-2k` (no audio is ever
downloaded -- only the text/metadata columns).

All commands below assume you're running from the repo root.

## Setup

```bash
python3.12 -m venv .venv   # 3.12, not 3.14 -- some deps aren't ready for 3.14 yet
source .venv/bin/activate
pip install -r requirements.txt
```

You need read access to the gated `ErfanRou/callcc-2k` dataset, and to any
gated model you train (e.g. `google/gemma-3-1b-it` requires an approved
access request at https://huggingface.co/google/gemma-3-1b-it first).
Authenticate one of two ways:

- copy `.env.example` to `.env` and fill in `HF_TOKEN` (checked first, works
  across all repo scripts), or
- `huggingface-cli login` (used as a fallback only if `.env`/`HF_TOKEN` isn't set).

These can differ -- huggingface_hub checks the `HF_TOKEN` environment
variable before the `huggingface-cli login` session cached on the machine, so
a token in `.env` always wins over a different account logged in via the CLI.

If you'll init a git repo here later, add `.env` and `.venv/` to `.gitignore`
before your first commit so the token never gets committed.

## Running on a GPU instance (e.g. Vast.ai)

`./run.sh` wraps venv setup, `.env`/HF auth, dataset build+curation, and
training into one entry point, and re-launches itself inside `tmux` so a
dropped SSH connection doesn't kill a multi-hour run:

```bash
./run.sh smoke   # default: tiny end-to-end check (see configs/smoke.yaml) --
                 # exercises train/save/hub-sync/test-eval/WER cheaply before
                 # committing to a real run. Do this first.
./run.sh build   # (re)build + curate the full training dataset
./run.sh sweep   # the 4-model comparison (configs/sweep.yaml) -- needs `build` first
./run.sh full    # build + sweep back to back -- multi-hour, real GPU cost
```

There's no git remote for this repo yet, so `run.sh` can't pull the code onto
the instance itself -- get the repo there first (`git clone` once you have a
remote, `rsync`, `scp`, or Vast's file upload), then run it from the repo
root. Everything below describes what each mode does step by step, for
running things manually/individually instead.

## Repo layout

```
run.sh                  entry point for a fresh GPU instance -- see above
configs/smoke.yaml      tiny end-to-end config used by `run.sh smoke`
build_dataset.py       (in src/) downloads + preprocesses callcc-2k into a
                        local .jsonl, column-pruned so audio is never fetched
src/prepare_split.py    curates build_dataset.py's output: drops misaligned
                        pairs, upsamples low-WER rows, caps assembled rows
src/data.py             dataset loading + tokenization/label-masking
src/config.py           YAML config -> Config dataclass, with typo checking
src/evaluate.py         runs the model on a test set, scores WER, saves
                        input/output/reference/WER per row
src/hub_sync.py         uploads a run's output_dir into its folder in the
                        shared Hub repo (checkpoints, tb logs, config, test_eval)
src/train.py            single training run (LoRA SFT)
src/run_sweep.py        runs several training configs back-to-back, then
                        compares them (by test WER, falling back to eval loss)
configs/base.yaml       template/example single-run config (not currently
                        used by sweep.yaml -- see below)
configs/qwen3.5-2b.yaml, configs/gemma-3-1b-it.yaml
                        standalone configs for the two models currently
                        being fine-tuned, each sized for their own memory
                        footprint rather than sharing base.yaml's
configs/sweep.yaml      currently queues just those two models' configs
docs/asr_dataset_builder.md
                        column schema + example rows for build_dataset.py's
                        output (chunked vs. assembled)
notebooks/load_model.ipynb
                        download one experiment's folder from the Hub and
                        run it on a sample transcript
```

## 1. Build the training data

```bash
python src/build_dataset.py --assembled-ratio 0.3 --output-dir ./asr_dataset.jsonl
```

Each call is randomly (but deterministically, via `--seed`) assigned to
either "assembled" rows (one row per channel, that channel's full turn
sequence joined into one example -- matching how `callcc-test-1k` and actual
serving are structured: one channel corrected at a time, not both speakers
merged) or "chunked" rows (one example per segment); `--assembled-ratio`
controls what fraction of calls become assembled. Use `--max-calls`/
`--max-shards` to generate a small sample first. See
[docs/asr_dataset_builder.md](docs/asr_dataset_builder.md) for the full
column schema and an example row of each type.

### Curate it before training

`src/prepare_split.py` fixes three issues found by inspecting an actual
sample of the raw output:

```bash
python src/prepare_split.py --input ./asr_dataset.jsonl --output ./asr_dataset_curated.jsonl
```

- **Drops likely-misaligned pairs** (`wer_whisper` above `--wer-cap`, default
  `1.0`): the highest-WER raw rows turned out to be cases where Whisper's
  segment window drifted onto a different utterance entirely, not a
  genuinely correctable ASR error. Training on those teaches hallucinated
  rewrites rather than correction.
- **Upsamples low-WER (`agree`-bucket) rows** to `--agree-target-frac`
  (default `0.225`) of the chunked rows: natural rate is only ~11%, and
  without enough "already correct, leave it" examples the model risks
  over-correcting fine transcripts in production.
- **Caps assembled (full-channel) rows** to `--assembled-target-frac` (default
  `0.10`) of the output -- downsampled if there are more than that, never
  artificially inflated if there are fewer.

Point `dataset_id` in your config at this curated file, not the raw one from
step 1.

## 2. Configure a run

Edit `configs/base.yaml` (or copy it) -- it's grouped into sections purely for
readability; `train.py` flattens them, and an unrecognized key (e.g. a typo)
raises an error immediately rather than being silently ignored:

- `model_id` / `dataset_id`: which model to fine-tune and which data to use.
  `dataset_id` can be a local path (like the `.jsonl` from step 1 -- in which
  case `eval_fraction` carves out a validation split, since that file has no
  predefined splits) or a Hub dataset repo id (in which case use
  `train_split`/`eval_split` instead).
- `max_length` / `on_long_example`: token budget for one tokenized example
  (prompt + target). Rows that don't fit are, by default
  (`on_long_example: drop`), excluded entirely rather than truncated --
  truncating cuts off the *end of the target*, training the model to produce
  a correction that just stops mid-sentence. Set to `truncate` to keep them
  anyway. Either way, a row whose *prompt alone* already exceeds
  `max_length` is always dropped, since there'd be no room left for any
  target token. `train.py` prints how many rows were kept/truncated/dropped
  after tokenizing.
- `training.*`: epochs, batch size, learning rate, save/eval cadence, etc.
- `lora.*`: LoRA rank/alpha/dropout.
- `quantization.load_in_4bit`: QLoRA; needs `bitsandbytes` + a CUDA GPU.
- `hub.*`: sync `output_dir` to the Hub as training runs (see "Where things
  end up" below) -- `repo_id` should be the **same shared repo across every
  config**, with `folder` (default: `output_dir`'s basename) giving each run
  its own subfolder inside it.
- `tensorboard.logging_dir`: where TensorBoard event files are written.
  Leave `null` to default to `<output_dir>/tb`.
- `test.*`: if `dataset_id` is set, the model is run against that test set at
  every checkpoint save, scored with WER, and the results saved under
  `output_dir` (see "Test-set evaluation" below). Leave `dataset_id: null` to
  skip this.
- `resume_from_checkpoint`: `false`, `true` (resume from the latest local
  checkpoint), a local checkpoint path, or `"hub"` to pull this run's folder
  from `hub.repo_id` first.

Whatever you put in a config, `train.py` writes the fully-resolved version to
`<output_dir>/config.yaml` at the start of every run -- this is what
`notebooks/load_model.ipynb` reads `model_id`/`system_prompt` back out of
later, and it's your record of exactly what produced a given checkpoint.

## 3. Train

```bash
python src/train.py --config configs/base.yaml
```

Override individual values without editing the file with repeatable `--set`:

```bash
python src/train.py --config configs/base.yaml \
  --set training.learning_rate=1e-4 \
  --set lora.r=32
```

### Monitor

```bash
tensorboard --logdir ./outputs   # or whatever tensorboard.logging_dir you set
```

### Resume a run

Set `resume_from_checkpoint: true` in the config (or `--set
resume_from_checkpoint=true`) and re-run the same command -- it picks up the
latest checkpoint under `output_dir`. To resume on a fresh machine (e.g. the
local `output_dir` was lost), set it to `"hub"` instead: this run's folder is
downloaded from `hub.repo_id` first, then resumed from the latest checkpoint
in it as normal.

### Where things end up

Every run's `output_dir` ends up self-contained:

```
outputs/qwen3-1.7b-base/
  config.yaml               the fully-resolved config that produced this run
  checkpoint-200/, checkpoint-400/, ...   pruned locally by save_total_limit
  best_checkpoint_wer/      best checkpoint by test WER seen so far, if test.dataset_id is set
  tb/                       TensorBoard event files
  test_eval/predictions.jsonl, metrics.json   latest test-set results, if test.dataset_id is set
  adapter_model.safetensors, tokenizer files, ...   the final saved adapter
```

If `hub.push_to_hub` is true, this whole folder is synced -- via
`src/hub_sync.py`, not Trainer's built-in `push_to_hub` (which assumes the
*entire* target repo is one run, and can't do "one shared repo, one folder
per run") -- to `hub.folder` inside `hub.repo_id` on every checkpoint save
and once more at the end. Checkpoints pruned locally by `save_total_limit`
are pruned from the Hub folder too, so it mirrors local state rather than
accumulating every checkpoint ever saved.

### Test-set evaluation

If `test.dataset_id` is set, every checkpoint save (and the final model)
triggers a generation pass over that dataset: the model corrects each
`test.input_column` value, and scored against `test.target_column`:

- `test_eval/predictions.jsonl` -- one row per example: `input`, `output`,
  `reference`, `wer`. Overwritten each time with the latest checkpoint's
  results (not one file per checkpoint).
- `test_eval/metrics.json` -- `{"wer": ..., "exact_match": ..., "n_examples": ...}`.
  `wer` (via `jiwer`, corpus-level) gives partial credit for near-misses;
  `exact_match` (fraction of rows the model got byte-for-byte right) is a
  stricter complementary read, and a direct signal on over/under-correction
  -- the same failure mode the system prompt and `prepare_split.py`'s
  agree-bucket upsampling are aimed at.
- both are also logged to TensorBoard as `test_wer`/`test_exact_match`, so
  you get curves over training steps, not just final numbers.

Use `test.max_examples` to cap the test set for faster per-checkpoint checks
if the full set is slow to run repeatedly.

### Best checkpoint by WER

Trainer's built-in `load_best_model_at_end`/`metric_for_best_model` only
tracks metrics from its own eval loop (`eval_loss`, computed on the training
data's held-out split) -- it has no way to know about `test_wer`, since
that's computed by a separate callback via actual generation on a different
dataset, not inside `Trainer.evaluate()`. So best-checkpoint tracking here is
custom, and deliberately tracks WER rather than loss, since WER against a
real test set is the metric that's mattered throughout this project, not
training loss.

Every checkpoint save (and the final model) compares its test WER against
whatever's recorded in `output_dir/best_checkpoint_wer/best_metrics.json`;
if it's better, that whole checkpoint gets copied there (overwriting the
previous best), tagged with its step and metrics. This is read back from
disk each time rather than kept in memory, so it stays correct across a
resumed run too. It only exists when `test.dataset_id` is set, and it's a
full independent copy -- pruning older numbered checkpoints via
`save_total_limit` never affects it.

## 4. Compare multiple models/configs

`configs/sweep.yaml` queues several jobs. Each job names its own `config`
file -- a job that only needs a different `model_id` could reuse
`configs/base.yaml` plus a one-line `overrides`, but every model actually in
the current queue (`qwen3.5-2b`, `gemma-3-1b-it`) gets its own standalone
config instead, since each has a different memory footprint and batch size
tuned for it (see "Notes" for GPU sizing). Each job's `output_dir`,
TensorBoard dir, and Hub *folder* are auto-namespaced by job name so they
never collide, even if the job's own config file sets those fields to
something else -- `hub.repo_id` is left alone, since every job is meant to
share the same repo.

`Qwen3-1.7B` and `gemma-4-E2B-it` aren't in the queue right now (on hold,
not removed) -- `configs/base.yaml` still exists as the single-run template/
example for `Qwen3-1.7B`, and a `gemma-4-E2B-it` config would need its own
smaller batch size (it's ~5B params on disk, the biggest of the four
candidates, despite the "E2B" name) before being added back.

```bash
python src/run_sweep.py --sweep configs/sweep.yaml
```

Jobs run one after another (they share a GPU) via `python src/train.py
--config <resolved config>` under the hood. A failed job doesn't stop the
queue -- it's recorded and the rest still run. At the end, a summary ranked
by test WER (falling back to eval loss for any job without `test.dataset_id`
set) is printed and written to `<output_root>/summary.json`.

## 5. Load a trained model

`notebooks/load_model.ipynb` downloads one experiment's folder from the Hub
(by `hub.repo_id` + `hub.folder`/`output_dir` basename), reads its
`config.yaml` to get `model_id`/`system_prompt` automatically, loads the base
model + LoRA adapter, and runs it on a sample transcript -- also shows that
run's `test_eval` results if it has any. Point `RUN_FOLDER` at a specific
checkpoint subfolder instead of the run root to load an earlier checkpoint
rather than the final saved adapter.

## Notes

- Training requires a CUDA GPU (LoRA/QLoRA on the four candidate models --
  two of which are multimodal `ConditionalGeneration` architectures that
  still register correctly under `AutoModelForCausalLM` in the pinned
  `transformers` version).
- `google/gemma-3-1b-it` is gated; make sure `HF_TOKEN` in `.env` belongs to
  an account with approved access (verified working for the PedramR account).
- Test evaluation runs a generation pass at every checkpoint save, which is
  slower than the training step itself -- use `test.max_examples` to keep
  per-checkpoint eval fast, or increase `training.save_steps` if it's
  dominating wall-clock time.
- **GPU sizing**: measured on-disk weight sizes (bf16), not the models'
  names, are what matter -- `gemma-3-1b-it` ~2GB, `Qwen3-1.7B` ~4GB,
  `Qwen3.5-2B` ~4.5GB, `gemma-4-E2B-it` ~10.25GB despite the "E2B" name
  (Google's elastic/MatFormer-style naming refers to *effective* compute,
  not stored params -- it's actually the biggest of the four). Since the
  sweep runs models one at a time, size for whichever model in the queue is
  largest, not the sum. A single 24GB GPU is the practical floor for LoRA +
  gradient checkpointing on these; 48GB removes the OOM risk on
  `gemma-4-E2B-it` entirely; 80GB-class cards (A100/H100) are overkill for
  LoRA on models this size. The `qwen3.5-2b.yaml`/`gemma-3-1b-it.yaml` batch
  sizes are sized as an estimate for a 32GB card (RTX 5090) -- unverified
  against real hardware, adjust based on `nvidia-smi` during your first run.
