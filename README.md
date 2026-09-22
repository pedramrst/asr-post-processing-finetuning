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

On a Vast.ai instance, skip this: `./run.sh` (below) reuses the image's
preconfigured `/venv/main` instead of creating `.venv`, so `torch` stays the
build already matched to that image's CUDA version rather than being
re-downloaded from a plain `pip install`.

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
./run.sh smoke                # default: tiny end-to-end check (see configs/smoke.yaml) --
                               # exercises train/save/hub-sync/test-eval/WER cheaply before
                               # committing to a real run. Do this first.
./run.sh build                 # (re)build + curate the full training dataset
./run.sh train qwen3.5-2b      # train just one config -- needs `build` first (see below)
./run.sh train gemma-3-1b-it   # same, for the other config
./run.sh sweep                 # the model comparison (configs/sweep.yaml) -- needs `build` first
./run.sh full                  # build + sweep back to back -- multi-hour, real GPU cost
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
                        pairs, upsamples low-WER rows, balances assembled vs
                        chunked rows, upsamples confirmed-named-entity rows
src/build_entity_eval_slice.py, src/build_typo_eval_slice.py
                        build the small held-out eval slices (named-entity,
                        typo/dictation-form) from callcc-test-1k -- see
                        "Named-entity eval slice"/"Typo/dictation-form eval
                        slice" below
src/data.py             dataset loading + tokenization/label-masking
src/config.py           YAML config -> Config dataclass, with typo checking
src/evaluate.py         runs the model on a test set, scores WER, saves
                        input/output/reference/WER per row
src/hub_sync.py         uploads a run's output_dir into its folder in the
                        shared Hub repo (checkpoints, tb logs, config, test_eval)
src/train.py            single training run (LoRA SFT)
src/run_sweep.py        runs several training configs back-to-back, then
                        compares them (by test WER, falling back to eval loss)
                        -- auto-continues the winner with a follow-up run
src/supervise.py        launches/monitors one training run as a detached
                        process, with CUDA-OOM auto-recovery -- see
                        "Telegram agent" below
src/tools.py            the ~20 named operations the Telegram agent can
                        call (process control, config, data/checkpoints,
                        results, sweeps, GPU/disk health)
src/notify.py           Telegram send/receive helpers
src/telegram_agent.py   the agent's long-polling loop (`./run.sh agent`)
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
notebooks/explore_finetuning_data.ipynb
                        browse build_dataset.py/prepare_split.py output by
                        hand -- composition, entity/typo detection, word
                        confidence, length distributions, sample/search tools
```

## 1. Build the training data

```bash
python src/build_dataset.py --assembled-ratio 0.9 --output-dir ./asr_dataset.jsonl
```

Each call is randomly (but deterministically, via `--seed`) assigned to
either "assembled" rows (one row per channel, that channel's full turn
sequence joined into one example -- matching how `callcc-test-1k` and actual
serving are structured: one channel corrected at a time, not both speakers
merged) or "chunked" rows (one example per segment); `--assembled-ratio`
controls what fraction of calls become assembled. Since production actually
corrects one full call channel at a time (confirmed, not just inferred from
`callcc-test-1k`'s row shape), this should normally be high -- assembled
rows are the primary training signal, not a minority. `prepare_split.py`'s
`--assembled-target-frac` (below) can only ever downsample assembled rows
relative to chunked ones, never manufacture more, so an insufficient supply
here can't be fixed downstream. Use `--max-calls`/`--max-shards` to generate
a small sample first. See
[docs/asr_dataset_builder.md](docs/asr_dataset_builder.md) for the full
column schema and an example row of each type.

### Curate it before training

`src/prepare_split.py` fixes three issues found by inspecting an actual
sample of the raw output:

```bash
python src/prepare_split.py --input ./asr_dataset.jsonl --output ./asr_dataset_curated.jsonl
```

- **Drops likely-misaligned pairs** (`wer_whisper` above `--wer-cap`, default
  `1.0`, *or* `overlap_pct` below `--overlap-floor`, default `30.0`): the
  highest-WER raw rows turned out to be cases where Whisper's segment window
  drifted onto a different utterance entirely, not a genuinely correctable
  ASR error. Training on those teaches hallucinated rewrites rather than
  correction. `overlap_pct` (word-overlap between `text_whisper` and the
  target) is a second, largely independent signal for the same failure mode
  -- WER is length-normalized and can stay misleadingly low on a misaligned
  pair, while overlap collapses toward zero whenever the two sides are
  actually about different content. `--overlap-floor`'s default is a
  starting point, not tuned against real data -- check the drop counts
  printed to stderr and adjust.
- **Upsamples low-WER (`agree`-bucket) rows** to `--agree-target-frac`
  (default `0.225`) of the chunked rows: natural rate is only ~11%, and
  without enough "already correct, leave it" examples the model risks
  over-correcting fine transcripts in production.
- **Balances assembled (full-channel) vs. chunked (per-segment) rows** to
  `--assembled-target-frac` (default `0.85`) of the output: assembled rows are
  the primary training signal, since production corrects one full channel at
  a time, so whichever side (assembled or chunked) is oversupplied relative
  to that ratio gets downsampled -- neither side is ever upsampled to hit it,
  so a target that the raw mix can't support falls back to whatever ratio the
  scarce side allows (printed to stderr as `achieved`).
- **Upsamples confirmed-named-entity rows** to `--entity-target-frac` (default
  `0.5`) of the output. Each real name/order-number appears in maybe one or
  two calls total, so without this the model sees very few gradient updates
  per entity relative to common dictation errors that repeat across hundreds
  of rows -- directly investigating fine-tuned predictions found reliable
  fixes for dictation-form errors (stutters, word-boundary merges, ZWNJ) but
  zero confirmed successes on an actually mis-heard name in a small sample.
  Detection combines two signals, either one qualifying a row (both only
  ever present on assembled rows): `build_dataset.py`'s `crm_context` (a row
  counts as "entity" if the CRM customer name or case-owner name has a word
  that actually appears in the target text -- a confirmed real entity, same
  as `build_entity_eval_slice.py`'s eval-slice detection), and its
  `word_confidence` (Soniox's own per-word confidence, from the `tokens/`
  shards) -- a word below `--entity-confidence-max` (default `0.45`) that's
  also long/rare enough (`--entity-confidence-max-freq`, default `3`) to
  plausibly be a name or hard domain term. The second signal matters
  because CRM data only covers ~25% of calls; confidence exists on every
  call. Getting a workable threshold took real tuning, not a guess:
  `build_dataset.py`'s storage cutoff (0.8) alone, even combined with
  length/rarity gates, still flagged 50-70% of assembled rows as "entity"
  at real-data (3,000-call) scale -- the original defaults (0.3/1) landed
  at ~16% there. Those defaults turned out too strict for real but
  moderately-rare domain vocabulary, though (verified directly: a
  fine-tuned model failed to correct a recurring product/scent term whose
  measured Soniox confidence, ~0.44, sat just above the 0.3 cutoff, so
  those rows never got the upsampling boost) -- hence the looser defaults.
  **Re-tune both against your own corpus before relying on them**: verified
  directly, `--entity-confidence-max-freq` has a sharp cliff around 4-5, not
  a gradual slope -- assembled rows are full-call length (often 100+
  words), so once the cap is loose enough, the odds that a long row
  contains *some* matching word by chance approach certainty, and the
  flagged share jumps from ~10-15% to 40-60%+ almost discontinuously.
  Flagged rates also differed drastically between a smaller local sample
  and the full-scale corpus these were originally tuned against -- don't
  assume a threshold transfers between corpora of very different size
  without checking the printed flagged-% first. `0` disables the whole
  step (both signals).
  `--entity-max-repeats` (default `5`) caps how many times any single row
  can be duplicated to get there -- verified directly: distinct customer
  names scale roughly linearly with how many raw calls you process (~324
  distinct names found in a real 6,000-call sample), but `--entity-target-
  frac` is a fraction of the whole output regardless of corpus size, so
  without a cap a small distinct pool relative to a large non-entity pool
  still gets repeated dozens of times each -- teaching the model to
  over-memorize a handful of specific people rather than the general skill.
  When the cap makes the target frac unreachable, the printed `achieved`
  falls honestly short of it rather than over-duplicating to compensate --
  process more raw calls (a higher `build_dataset.py --max-calls`, or none
  at all) for a larger, more diverse distinct-entity pool instead of raising
  the cap.

Point `dataset_id` in your config at this curated file, not the raw one from
step 1.

To manually inspect either file -- composition, entity/typo detection, word
confidence, length distributions, or just browsing/searching real rows --
open `notebooks/explore_finetuning_data.ipynb` and point it at your own
`build_dataset.py`/`prepare_split.py` output (defaults to the sample paths
used to build it).

## 2. Configure a run

Edit `configs/base.yaml` (or copy it) -- it's grouped into sections purely for
readability; `train.py` flattens them, and an unrecognized key (e.g. a typo)
raises an error immediately rather than being silently ignored:

- `model_id` / `dataset_id`: which model to fine-tune and which data to use.
  `dataset_id` can be a local path (like the `.jsonl` from step 1 -- in which
  case `eval_fraction` carves out a validation split, since that file has no
  predefined splits) or a Hub dataset repo id (in which case use
  `train_split`/`eval_split` instead).
- `train_fraction`: deterministically subsamples only the train split to this
  fraction of its rows (`eval`/`validation` stays full-size). `null`
  (default) or `1.0` uses every row. Meant for a data-scaling ablation --
  e.g. train on 10%/25%/50%/100% of the curated data and compare
  `eval_loss`/test WER -- without hand-rolling a separate curated file per
  fraction: `--set train_fraction=0.1`.
- `max_length` / `on_long_example`: token budget for one tokenized example
  (prompt + target). Rows that don't fit are, by default
  (`on_long_example: drop`), excluded entirely rather than truncated --
  truncating cuts off the *end of the target*, training the model to produce
  a correction that just stops mid-sentence. Set to `truncate` to keep them
  anyway. Either way, a row whose *prompt alone* already exceeds
  `max_length` is always dropped, since there'd be no room left for any
  target token. `train.py` prints how many rows were kept/truncated/dropped
  after tokenizing -- worth checking after raising `prepare_split.py`'s
  `--assembled-target-frac`, since assembled (full-channel) rows are much
  longer than chunked ones and a higher assembled share means more of them
  can exceed `max_length` and get dropped exactly where they matter most.
- `mask_low_signal_corrections` / `mask_min_similarity` /
  `mask_max_common_freq`: opt-in feature flag to mask un-guessable
  corrections out of the loss -- see "Low-signal correction masking" below.
- `use_unsloth`: opt-in feature flag to load the model/set up LoRA via
  Unsloth instead of plain transformers + peft -- see "Unsloth" below.
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

For a quick run on a fraction of the curated data (e.g. as part of a
data-scaling ablation), also override `output_dir`/`hub.folder` so it doesn't
collide with a full run's:

```bash
python src/train.py --config configs/qwen3.5-2b.yaml \
  --set train_fraction=0.1 \
  --set output_dir=./outputs/qwen3.5-2b-10pct \
  --set hub.folder=qwen3.5-2b-10pct
```

The same works through `./run.sh train <config>` -- anything after the
config name is forwarded to `train.py` as-is:

```bash
./run.sh train qwen3.5-2b --set train_fraction=0.1 --set output_dir=./outputs/qwen3.5-2b-10pct
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
  `reference`, `wer`, `input_overlap_pct`, `hallucinated`. Overwritten each
  time with the latest checkpoint's results (not one file per checkpoint).
- `test_eval/metrics.json` -- `{"wer": ..., "exact_match": ..., "hallucination_rate": ..., "n_examples": ...}`.
  `wer` (via `jiwer`, corpus-level) gives partial credit for near-misses;
  `exact_match` (fraction of rows the model got byte-for-byte right) is a
  stricter complementary read, and a direct signal on over/under-correction
  -- the same failure mode the system prompt and `prepare_split.py`'s
  agree-bucket upsampling are aimed at.
- `hallucination_rate` / `hallucinated` / `input_overlap_pct` are a separate
  axis from `wer`/`exact_match`: those two only compare the output against
  the *reference*, so they can't distinguish "wrong correction" from
  "invented content ungrounded in the input." `input_overlap_pct` is the
  share of the *output*'s words that actually appear in the *input*; a row
  is flagged `hallucinated` when that's below `test.hallucination_overlap_floor`
  (default `50.0`). This is the model's actual generation behavior, a
  complement to `prepare_split.py`'s `--overlap-floor` (below), which instead
  filters *training* pairs before the model ever sees them.
- all three (`wer`, `exact_match`, `hallucination_rate`) are also logged to
  TensorBoard as `test_wer`/`test_exact_match`/`test_hallucination_rate`, so
  you get curves over training steps, not just final numbers.

`test.repetition_penalty` (default `1.2`) and `test.no_repeat_ngram_size`
(default `3`) guard against a real, observed failure of plain greedy
decoding (`do_sample=False`): a short, locally-high-probability phrase --
e.g. this task's "بله" (yes) agreement-word bursts, which do occur
naturally as short runs in real training segments -- can trigger a runaway
repeat loop that never finds the actual stopping point and just fills
`max_new_tokens`. On this task's first fine-tuned checkpoints, this hit
roughly 1 in 6-7 test examples and single-handedly dragged the aggregate
`wer`/`hallucination_rate` far worse than the untrained baseline, even
though the ~85% of predictions that didn't degenerate were good corrections.
Set either to `1.0`/`0` to disable.

`test.max_examples` caps the *baseline* (below) and *final* evals -- each
only runs once per training run, so leaving it `null` (the full test set) is
usually worth the time for an accurate number. `test.checkpoint_max_examples`
is a separate, usually much smaller cap for the *repeated* per-checkpoint
eval: with a frequent `eval_steps`/`save_steps` that can fire hundreds of
times over a run, so running the full set there every time multiplies eval
cost far beyond training itself -- both `qwen3.5-2b.yaml`/`gemma-3-1b-it.yaml`
set this to `150`. `null` (the default) falls back to `test.max_examples`
(no separate cap).

Generation is also the one place `test.batch_size` (default `8`, raised to
`16` in the two active configs) matters independently of training's own
batch size -- and unlike a standalone `evaluate.py` CLI run, the
per-checkpoint eval runs *inside* the training process's existing GPU
footprint (LoRA + optimizer state already resident), so watch for OOM
specifically during a checkpoint save, not just training steps, before
pushing it higher.

This isn't just a theoretical caveat: with `batch_size: 16`, generation's
KV-cache churn (allocating/freeing many differently-sized tensors as
sequences grow token-by-token) fragmented PyTorch's CUDA allocator badly
enough to OOM the *training* step right after the first checkpoint eval,
reproduced identically on two separate runs on a 32GB RTX 5090 -- not a raw
capacity problem (the error showed several GiB "reserved but unallocated"
that a single large backward-pass allocation couldn't use). `run_test_eval()`
now calls `torch.cuda.empty_cache()` right after generation to release that
back before training resumes, and `run.sh` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as further mitigation. If
OOM recurs anyway, `test.batch_size` is the first thing to dial back down.

Generation batches are formed by length (longest first), not dataset order --
`model.generate()` only finishes a batch once every sequence in it has
stopped, so one long input landing next to several short ones used to force
the whole batch to keep decoding (and padding) far past where the short ones
would've finished alone. Given this task's huge length spread (chunked rows
~15-40 words vs. assembled rows up to ~3,000), an unlucky batch composition
could single-handedly dominate an eval pass's wall-clock. `generate_batch()`
sorts prompts by length before batching and restores the original order
before returning, at no cost beyond the sort itself (verified with a unit
test that output-to-input correspondence survives the reordering correctly).
Training batches get the same treatment via `train_sampling_strategy:
"group_by_length"` in `TrainingArguments` (the successor to the old
`group_by_length: true` boolean flag, renamed in the transformers version
this repo pins -- verified directly, the old kwarg raises `TypeError` here).

If a model's `transformers` warns about `causal_conv1d`/
`chunk_gated_delta_rule` falling back to a reference PyTorch implementation
(seen with Qwen3.5's hybrid SSM/linear-attention layers), that slows down
both training and every generation pass here. `run.sh` best-effort installs
`causal-conv1d`/`flash-linear-attention` to fix it (see `requirements.txt`)
-- best-effort because they're CUDA/torch-version-sensitive compiled
extensions, not guaranteed to build on every image, so a failure there
doesn't abort the rest of the script.

### Baseline (pre-fine-tuning) evaluation

If `test.dataset_id` is set and `test.baseline` is true (the default), one
extra eval runs *before the first training step*, saved separately under
`test_eval_baseline/predictions.jsonl` + `metrics.json` and logged to
TensorBoard at step 0 -- so the `test_wer`/`test_exact_match`/
`test_hallucination_rate` curves show where the model started, not just the
first checkpoint. This doesn't need a separate unadapted-model load: LoRA's
`B` matrix is zero-initialized, so the freshly-wrapped, untrained model is
numerically identical to the plain base model at that point. Skipped
automatically on a resumed run (`resume_from_checkpoint` set) -- the original
run already captured this -- or set `test.baseline: false` to skip it
otherwise (e.g. re-running several `train_fraction` ablations of the same
model, where the baseline would be identical every time).

### Named-entity eval slice

Aggregate WER on the full test set mixes entity-heavy and entity-free calls
together, which hides exactly the failure mode this task's system prompt
cares about most -- person/place/order-name correction -- inside a number
dominated by everything else (found by direct investigation: on this task's
first fine-tuned checkpoints, dozens of "rare word" fixes turned out on
inspection to be dictation-form corrections like word-boundary merges or
stutters, not actual named-entity corrections, while several confirmed real
name mishearings -- e.g. an agent's own name misheard as a filler phrase --
went uncorrected).

```bash
python src/build_entity_eval_slice.py --output data/entity_eval_slice.jsonl
```

`callcc-test-1k` carries `crm_metadata` (the real CRM record) on every row.
This cross-references the customer's real name and the CRM record's
`owner_name` (usually the agent, in Persian -- `crm.agent.name` is Latin
script and won't match transcript text) against the reference transcript,
keeping only rows where a confirmed real name-word actually appears in it --
so you know for certain a genuine named entity is in play, not a rare-word
heuristic guess. Point `test.entity_dataset_id` at the output to run it
alongside the main test set at baseline/every checkpoint/final, reusing
every other `test.*` generation setting: results land under
`test_eval_entity/` (`test_eval_entity_baseline/` for the pre-training pass)
and log to TensorBoard as `test_entity_wer`/`test_entity_wer_zwnj_normalized`/
`test_entity_exact_match`/`test_entity_hallucination_rate`, tracked
separately from the main `test_*` curves throughout training.

### Typo/dictation-form eval slice

Complements the entity slice with the *other* main error category --
stutters, word-boundary merges, ZWNJ half-spacing, letter-level slips --
which direct investigation found the model already handles well. This is a
**regression guard**, not an improvement target: as training leans harder on
entity correction (`prepare_split.py`'s entity upsampling), this is what
would catch it if that quietly erodes the dictation-error correction that
already works.

```bash
python src/build_typo_eval_slice.py --output data/typo_eval_slice.jsonl
```

Same word-alignment method as the entity slice, opposite signal: for each
reference word Whisper got wrong, high character overlap with Whisper's
word (ZWNJ-normalized) means a spacing/stutter/boundary slip, low overlap
means a different word entirely (an entity slice concern, not this one). A
row qualifies only if it has a typo-like error and *no* entity-like error,
so the two slices stay disjoint. Point `test.typo_dataset_id` at the output
-- same wiring as the entity slice (`test_eval_typo/`,
`test_typo_wer`/`test_typo_wer_zwnj_normalized`/`test_typo_exact_match`/
`test_typo_hallucination_rate`), and both run through the same
`run_secondary_eval()` helper in `evaluate.py`, so adding another named
slice beyond these two doesn't mean copy-pasting a third near-identical
eval block into `train.py`/`TestEvalCallback`.

### Punctuation

By default this task is punctuation-free end to end: `text_whisper` (this
model's actual input) never has punctuation, `text` (the default
`target_column`) has had it stripped, and the system prompt says so
explicitly ("Do not add punctuation"). But every dataset this pipeline
touches also carries a *punctuated* version of the same reference --
`text_soniox` in our own curated data, `text_raw` on
`ErfanRou/callcc-test-1k` and the entity/typo eval slices -- currently
unused. `include_punctuation: true` swaps in a punctuation-aware system
prompt (`data.py`'s `SYSTEM_PROMPT_WITH_PUNCTUATION`) so you can test
training/evaluating against it instead.

It's deliberately *not* a single do-everything flag: it only swaps the
system prompt. You still set `target_column`/`test_target_column`
explicitly, since the punctuated column has a different name per dataset
and guessing wrong would silently train on the wrong target with no error.
`train.py` warns at startup (doesn't block) if the flag and the column
names it recognizes look inconsistent with each other:

```bash
python src/train.py --config configs/qwen3.5-2b.yaml \
  --set include_punctuation=true \
  --set target_column=text_soniox \
  --set test.target_column=text_raw \
  --set output_dir=./outputs/qwen3.5-2b-punctuated
```

Worth knowing before you compare results: this is a strictly harder version
of the task, not just an added nicety. Whisper's output carries no acoustic
pause/prosody cues, so the model has to infer sentence structure from
content alone. `wer`/`wer_zwnj_normalized` also aren't directly comparable
across punctuated vs. punctuation-free runs -- a word-level WER treats
`"سلام،"` and `"سلام"` as different tokens, so every sentence-boundary guess
the model gets even slightly wrong counts as a full word error on top of
whatever content/entity errors it also made.

### Low-signal correction masking

Some of what's in the training targets is genuinely un-guessable from what
Whisper produced: a name mis-heard as a completely unrelated word, with no
phonetic or character-level overlap to recover from. Training on those rows
as-is means the loss rewards the model for memorizing the correct answer for
that specific call, but for a *new* call with the same kind of failure,
there's no signal in the input pointing to the fix -- the model can only
guess, and training it to output a specific guess anyway just teaches it to
guess confidently. That's a property of Whisper's failure on that word, not
something worth reinforcing.

`mask_low_signal_corrections: true` masks those spans out of the loss
(`labels = -100`) at the token level, leaving everything else --
correct passthrough text and *guessable* corrections alike -- training
normally. "Guessable" is decided by `data.py`'s `low_signal_word_spans()`:
for each word-level alignment error between `whisper_text` and the target,
compute a phonetic similarity between what Whisper said and the correct
word, normalizing well-documented Persian homophone letter groups first
(ز/ذ/ض/ظ, س/ص/ث, ت/ط, ق/غ, ح/ه -- deliberately not ک/گ, which aren't true
homophones) so a real homophone slip isn't penalized just for sharing few
raw characters with the correction. A word Whisper produced *nothing* for
at all (a "delete"-type alignment chunk) has no Whisper-side content to
compare against, so it's always treated as unguessable regardless of
`mask_min_similarity` -- **unless** the frequency gate below rules it out.

Phonetic similarity alone isn't enough, though: tested directly against the
curated training set, it flagged mostly common function/filler words
("رو", "بله", "و", "هم", "خب"), not names -- a dropped "بله" (yes) scores
the same 0.0 similarity as a dropped name, even though it's trivially
predictable from Persian dialogue structure regardless of what Whisper
produced. `low_signal_word_spans()` also takes a corpus-wide word-frequency
count (`train.py` builds this once over every training target before
tokenizing) and only masks a span if at least one of its words occurs at
most `mask_max_common_freq` times in it -- a span made up entirely of common
words is left in the loss. Verified on an 800-row sample: this cut the
share of examples with any masking from 89% to 52% and total masked tokens
by ~83%, and the masked spans left afterward were genuinely names
("شاهرخی", "صفایی‌فر", "حسن‌پور") and rare content words ("مالیات",
"مصاحبه", "سومین"), not routine dictation-error vocabulary.

```bash
python src/train.py --config configs/qwen3.5-2b.yaml \
  --set mask_low_signal_corrections=true \
  --set output_dir=./outputs/qwen3.5-2b-masked
```

`mask_min_similarity` (default `0.75`) controls the phonetic cutoff --
tuned on a small hand-checked sample to cleanly separate known-guessable
pairs (>= 0.833: homophone substitutions, ZWNJ/spacing-only differences)
from known-unguessable ones (<= 0.714); treat it as a starting point, not a
precise threshold. `mask_max_common_freq` (default `1`) is the frequency
gate's cutoff -- a word occurring at most this many times across the whole
training set counts as rare. `train.py` prints how many target tokens got
masked out this way after tokenizing, alongside the existing
kept/truncated/dropped counts.

The token-to-span mapping is done via the tokenizer's offset mapping,
locating the target's token ids as a subsequence inside the full tokenized
example rather than assuming they start immediately after the prompt --
some chat templates insert boilerplate there (e.g. Qwen3's template adds an
empty `<think>\n\n</think>\n\n` block even when the message itself has no
thinking content). If that subsequence search fails for some row (observed
only for a row truncated mid-target), masking is silently skipped for that
row rather than approximated -- the row still trains normally, just without
this extra masking.

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

### Unsloth

`use_unsloth: true` swaps model loading and LoRA setup over to
[Unsloth](https://unsloth.ai)'s `FastLanguageModel`, in place of plain
`transformers.AutoModelForCausalLM` + `peft.get_peft_model`. Unsloth patches
the model with fused kernels and its own gradient-checkpointing
implementation, and reports (their own benchmarks, not independently
re-verified here) roughly 2x faster training and ~70% less VRAM on the
Qwen3/Gemma3 model families this repo uses. Its optimization is applied at
the model-loading step, not the training loop, so it's compatible with the
plain `transformers.Trainer` this script already uses (not TRL's
`SFTTrainer`, which every one of Unsloth's own quickstart examples happens
to use, but isn't required) -- everything else (the custom `PadCollator`,
`TestEvalCallback`, hub sync, baseline eval, masking) is unaffected.

```bash
python src/train.py --config configs/qwen3.5-2b.yaml --set use_unsloth=true
```

Requires a CUDA GPU and the `unsloth` package (see `requirements.txt`).
**Not verified end-to-end in this repo** -- there is no CUDA GPU in the
environment this integration was written and reviewed in, so unlike
everything else in this pipeline, this specific path could only be checked
against Unsloth's own documented API, not actually run. Test it on a real
GPU (a short run, not a full fine-tune) before trusting it. A few interacting
settings to be aware of, verified by reading rather than executing:
- `load_in_4bit` is passed straight to `FastLanguageModel.from_pretrained`
  instead of building a separate `BitsAndBytesConfig` -- don't expect both
  paths to be active at once.
- `gradient_checkpointing` in `TrainingArguments` is automatically forced off
  when `use_unsloth` is on (`get_peft_model(..., use_gradient_checkpointing=
  "unsloth")` already handles it) -- leaving both on would double up two
  different checkpointing implementations rather than complementing each
  other.
- `target_modules` uses the explicit 7-name projection list from Unsloth's
  own docs (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/
  `down_proj`), not this repo's usual `"all-linear"` shorthand, since
  Unsloth's `get_peft_model` doesn't accept that shorthand.

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

To train just one of these configs instead of the whole queue, use
`./run.sh train <name>` (e.g. `./run.sh train qwen3.5-2b`) or call
`python src/train.py --config configs/qwen3.5-2b.yaml` directly. Either way,
`output_dir`/`hub.folder` come straight from that config file (e.g.
`./outputs/qwen3.5-2b`) rather than being namespaced under
`sweep.yaml`'s `output_root` the way a `run_sweep.py` run would.

## 5. Load a trained model

`notebooks/load_model.ipynb` downloads one experiment's folder from the Hub
(by `hub.repo_id` + `hub.folder`/`output_dir` basename), reads its
`config.yaml` to get `model_id`/`system_prompt` automatically, loads the base
model + LoRA adapter, and runs it on a sample transcript -- also shows that
run's `test_eval` results if it has any. Point `RUN_FOLDER` at a specific
checkpoint subfolder instead of the run root to load an earlier checkpoint
rather than the final saved adapter.

## 6. Telegram agent

`./run.sh agent` starts a long-running process that lets you control this
pipeline from Telegram in plain, technical language -- e.g. "what's the WER
trend on the qwen run", "drop batch size to 1 and restart", "show me the
highest-WER predictions from the latest checkpoint". It's built on LLM
tool-calling, not a keyword classifier -- several requests need free-form
argument extraction from casual phrasing (which config, which field, what
value; which checkpoint; what filter), which is a generation problem, not a
fixed-label one; tool-calling handles intent selection and argument
extraction in one call.

Needs `OPENROUTER_API_KEY` in `.env`, routed through OpenRouter's universal
OpenAI-compatible chat completions endpoint (`src/telegram_agent.py`'s
`_make_client()`) -- deliberately *not* Anthropic's own tool_runner/
OpenRouter's Anthropic-specific endpoint, which only works for Anthropic
models: using the OpenAI-format tool-calling loop instead means `AGENT_MODEL`
(default `anthropic/claude-opus-5`) can be set to any tool-calling-capable
model OpenRouter carries -- e.g. `openai/gpt-5`, `google/gemini-3-pro` -- see
https://openrouter.ai/models. This required no changes to `tools.py`'s tool
definitions: OpenAI's `parameters` field and Anthropic's `input_schema` are
both plain JSON Schema, so `telegram_agent.py` just reshapes the
`@beta_tool`-generated schemas rather than redefining them, and calls the
underlying function directly via each tool's `.func`.
extraction in one call.

Deliberately **not** an open "run arbitrary shell/Python" tool. Every
capability the agent has is one of a fixed, named set of Python functions in
`src/tools.py` -- what it can do is bounded and auditable, not "whatever it
decides to type":

| Group | Tools |
|---|---|
| Process control | `check_training_status`, `stop_training`, `run_finetune`, `resume_training` |
| Config | `edit_config`, `get_effective_config`, `validate_config` |
| Data & checkpoints | `build_data`, `list_checkpoints`, `check_hf_upload`, `sync_to_hub` |
| Results & metrics | `get_checkpoint_metrics`, `get_secondary_eval_metrics`, `sample_predictions`, `query_predictions`, `compare_runs` |
| Sweeps | `run_sweep` |
| Operational health | `check_gpu`, `check_disk_usage`, `tail_log` |

`query_predictions`'s `filter` argument is a small fixed-vocabulary DSL
(`{"field": ..., "op": ..., "value": ...}`, restricted to known prediction
fields and comparison operators) -- not an arbitrary expression or code
string. This is the one place a careless design could reintroduce
uncontrolled code execution; keeping it a constrained mini-DSL
(`tools.apply_prediction_filter`, unit-tested) is deliberate.

### Confirmation

State-changing tools (`stop_training`, `run_finetune`, `resume_training`,
`edit_config`, `build_data`, `sync_to_hub`, `run_sweep`) take a `confirmed`
parameter. There's no separate pending-action state machine: the gate lives
*inside the tool function itself* -- called with `confirmed=False` (the
default), it describes what it would do without doing it, and the model's
reply naturally asks you to confirm. Your next message is just the next
turn in the same conversation, and the model decides from context whether
it's a confirmation, a cancellation, or something unrelated -- the same way any
other multi-turn tool use works, not a hand-written yes/no keyword parser.

### The training supervisor (`src/supervise.py`)

Backs `run_finetune`/`resume_training`/`stop_training`. Launches `train.py`
as a **detached** background process (so the agent stays responsive to new
messages while training runs for hours) and monitors it for a CUDA-OOM
crash specifically -- nothing broader. On a match, it halves
`per_device_train_batch_size` (floor 1) and doubles
`gradient_accumulation_steps` to preserve the effective batch size, halves
`test.batch_size` too (the one OOM already seen in this project happened
during checkpoint-eval generation, not a training step), sets
`resume_from_checkpoint`, and retries -- capped, and never by editing the
checked-in YAML (every adjustment is an in-memory `--set`-style override the
supervisor tracks itself). Any other failure -- including a silent
host-level (CPU RAM) OOM, which the Linux kernel SIGKILLs with no traceback
at all -- stops and sends a Telegram alert instead of guessing at a fix. A
deliberate stop (SIGINT/SIGTERM to the supervisor) is distinguished from a
crash, so Ctrl-C in the tmux pane doesn't trigger a false crash alert.

State (pid, config, output_dir, retry count) is persisted to
`.agent_training_state.json` so a restarted agent (or a `check_training_status`
call from a fresh process) can always answer "is something running" without
holding an in-memory handle across process boundaries. A running supervisor
also periodically checks `test_eval/metrics.json` and sends a checkpoint
digest (WER, exact_match, hallucination_rate, and a trend classification --
improving/plateaued/regressed -- computed from its own
`supervisor_wer_history.jsonl`, since `best_checkpoint_wer/best_metrics.json`
only ever updates on improvement and so can't show a plateau or a
regression on its own).

### Sweeps auto-continue their winner

After `run_sweep.py` ranks jobs by test WER (unchanged from before), it now
also automatically launches a longer follow-up run from the winning job's
own resolved config (`<output_root>/<name>/config.yaml`, not the raw
`sweep.yaml` job entry, which lacks that job's specific overrides), resumed
from that job's `best_checkpoint_wer/` into a **new** output directory
(`<name>-followup`, so the follow-up's artifacts don't overwrite the
original sweep entry's results in place) -- routed through the same
supervisor, so it gets the same crash recovery as any other run. Explicitly
guards against an all-failed sweep (the ranking's `sorted(...)[0]` would
otherwise still return a job even when every `test_wer` is `None`).

### Caveats

Not verified end-to-end against a real GPU/CUDA OOM or a live Telegram/
OpenRouter API round-trip in the environment this was built in (no CUDA
GPU, no OpenRouter/Telegram credentials there) -- the supervisor's orchestration logic
(launch/detect/relaunch/cap-retries/SIGTERM-handling) was verified instead
against a fake stand-in training script that can be told to raise a
synthetic OOM, exit cleanly, or fail with an unrelated error, which exercises
the same code paths without needing a GPU. Test this on the real instance
(and confirm the actual OOM log signature matches a real traceback) before
relying on it unattended.

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
