# Edit-based ASR correction

A second correction method living next to the full-rewrite LoRA SFT
pipeline in `src/` (which it doesn't modify). Instead of regenerating the
whole transcript, a corrector proposes **word-level edits** against the
Whisper text; deterministic code validates and applies them, so every word
outside an accepted edit is Whisper's own. An empty edit list leaves the
transcript unchanged.

Goal: **fix real recognition errors and entities**, not match Soniox's
spelling/dialect choices. Background: `ASR_Edit_Based_Correction_Handoff.md`
at the repo root.

## Modules

| file | what |
|---|---|
| `edits.py` | `Edit` (word-index span + replacement), `extract_edits` (any source/hypothesis pair -> minimal edits, via jiwer), `validate_edits`, `apply_edits` |
| `scoring.py` | `score()`: evaluate.py's metrics computed identically (reproduces a run's `metrics.json`) plus fixed/broken counts, broken per 100 correct words, rows changed, entity fix/keep rates |
| `grounding.py` | inference-time CRM name candidates, `crm_snap_edits` (model-free name snapping), eval-only reference entity labels (CRM + Gemini extractions) |
| `rules.py` | model-free fixes for glued words and stutter-doubled letters |
| `diff_filter_baseline.py` | step 1 (below) |
| `build_diffs.py`, `llm_judge.py`, `make_label_sheet.py`, `calibrate.py`, `generate_targets.py` | step 2: differences, LLM judges, hand-labelling pages, judge calibration, training windows |
| `edit_format.py`, `windowing.py`, `train_edit.py`, `publish_edit_dataset.py`, `decode_sweep.py` | step 3: the model's input/output format, whole-transcript inference over overlapping windows, training + evaluation, publishing the training windows |
| `corpus.py`, `normalize.py` | shared helpers: corpus word frequencies; punctuation stripping and style-only detection |

Scripts are run from the repo root, e.g. `python3 src/edit/diff_filter_baseline.py`.
Outputs go under `outputs/edit/` (gitignored).

## Step 1: no-training baseline (`diff_filter_baseline.py`)

Takes the **original, not-fine-tuned Qwen3.5-2b**'s test predictions
(`qwen3.5-2b-50pct-masked-weighted/test_eval*_baseline` on the Hub), turns
each rewrite into edits against Whisper, keeps only the edits passing a
filter, and re-scores. Adds model-free methods (`rules`, `crm_snap`) for
comparison. No GPU needed; takes about 5 minutes the first time (it counts
corpus word frequencies, then caches them).

Results (main test set, 1,978 rows; lenient = persian_normalize-equivalent
forms count as correct; "broken/100" = new errors per 100 words Whisper had
right):

| method | WER | fixed | broken | net | broken/100 | entity fix |
|---|---|---|---|---|---|---|
| whisper (untouched) | 0.193 | 0 | 0 | 0 | 0.00 | 0.000 |
| original Qwen rewrite | 0.267 | 1,828 | 26,785 | −24,957 | 8.92 | 0.007 |
| its edits, punctuation stripped | 0.234 | 2,178 | 16,480 | −14,302 | 5.49 | 0.008 |
| its edits, best phonetic filter (0.8) | 0.193 | 221 | 388 | −167 | 0.13 | 0.003 |
| its edits, **oracle** filter | 0.188 | 2,062 | 13 | +2,049 | 0.00 | 0.008 |
| `crm_snap` | 0.193 | 21 | 27 | −6 | 0.01 | 0.023 |
| **`rules`** | **0.186** | 3,473 | 109 | **+3,364** | 0.04 | 0.007 |
| `rules` + `crm_snap` | 0.186 | 3,494 | 136 | +3,358 | 0.05 | 0.030 |

The entity and typo slices show the same pattern. Full tables:
`outputs/edit/diff_filter_baseline/summary.md`.

What it showed:

1. **Filtering the base model's rewrites isn't worth pursuing.** Even a
   perfect (oracle) filter only gets WER 0.193 -> 0.188, and every
   realistic filter is net-negative. The model's harmful edits are mostly
   colloquial -> formal rewrites (`باشه` -> `باشد`, `هیچ‌کدوم` -> `هیچ‌کدام`)
   and suffix drops (`سفارشی` -> `سفارش`); they sound alike, so a phonetic
   gate can't separate them from real fixes. About half of all broken words
   come from 67 rewrites that were cut off at the eval's 256-token
   generation limit.
2. **The biggest easy win is mechanical, not linguistic.** Most helpful
   edits were glued words where Whisper segments were joined
   (`بزنیبله` -> `بزنید بله`, `نکنهخواهش` -> `نکنه خواهش`) and doubled first
   letters (`اارسال`). `rules.py` fixes these deterministically and beats
   the oracle: +3,364 net words at 0.04 broken per 100. These errors don't
   need a model; they should be handled before (or instead of) one.
   Remaining false splits are real words that happen to be two common words
   (`میزنیم` -> `میز نیم`).
3. **The reference is unreliable for names.** When `crm_snap` corrects a
   misrecognized name to its CRM spelling, the Soniox reference contains
   that CRM spelling only 18/59 times. It keeps Whisper's form 8 times and
   has a third spelling 33 times. So the metric can't credit most correct
   name fixes, and training on Soniox text would teach wrong entity
   spellings. This supports verifying reference corrections (step 2) and
   treating CRM spellings as authoritative.
4. The CRM record only gives person names (customer, case owner). There's
   no product/brand catalog, so products/brands are evaluated (via Gemini
   labels) but not yet grounded.

## Step 2: clean training targets

Soniox's `text` is itself ASR output, so its differences from Whisper are
checked one at a time before any of them become correction targets.

```bash
python3 src/edit/build_diffs.py --max-rows 5000   # rows -> differences (outputs/edit/targets/)
python3 src/edit/make_label_sheet.py              # 200-item calibration sample + labelling page
open outputs/edit/targets/label_calibration.html  # label by hand, Export -> calibration_labels.json
python3 src/edit/llm_judge.py --items outputs/edit/targets/calibration_items.jsonl
python3 src/edit/calibrate.py                     # judge vs hand labels -> calibration_report.md
```

- `build_diffs.py` applies `rules.py` first, aligns Whisper against Soniox,
  marks style-only differences (not judged), and sorts the rest into
  strata: `entity`, `phonetic_sub`, `other_sub`, `whisper_dropped`,
  `whisper_extra`, `long` (plus the never-judged `filler` and `garbled`,
  see round 1 below). On a 5,000-row sample: 94.7k differences, 59.3k of
  them judgeable (about 12 per row), so judging the full 114k-row corpus
  isn't practical. Training windows get selected first (below).
- `llm_judge.py` shows each difference **blind**: the two versions as A/B
  in a per-item random order, so neither a judge nor the person labelling
  knows which is Soniox. Verdicts map to `real_error` (Soniox right),
  `soniox_wrong`, `either_fine`, `both_wrong`, `uncertain`. Each model
  runs as `@think` (reasoning on) and `@nothink`; results are cached per
  variant in `outputs/edit/judge/`.
- Which judge to trust is decided by `calibrate.py` against the hand
  labels, mainly by **real_error precision** (a wrong target teaches
  over-editing; a missed one just leaves Whisper's text).

### Hand-labelling round 1 and what changed

The first 200 items were often undecidable from text, and the prompt v1 /
page had no answer for "both are fine". The person used "same" for that
(filler words, small variants that both fit), while the judges read "same"
as spelling-only and forced an A/B choice. Judges' real_error precision was
57–65%. Changes:

- `build_diffs.py` now splits spans mixing a style-only and a real
  difference (`نمونه‌پور بله یه` vs `نمونه پور`), and never judges `filler`
  (only filler words on both sides) or `garbled` (2+ other differences
  within 3 words, so the context is unreadable). Check against the round-1
  labels: filler items were 14/21 "doesn't matter", style 9/10; the few
  real errors dropped just stay Whisper's text.
- Prompt v2: A/B only when one version is clearly right and the other
  clearly wrong; new `either` verdict for "both acceptable"; the context is
  flagged as possibly wrong itself. The page has the same options plus a
  separate note field.
- `make_label_sheet.py --refresh` re-maps labelled items onto a rebuilt
  `diffs.jsonl` (139 of 200 kept their labels); `--review` builds a page
  with only the items where the judges agree with each other but not with
  the person, plus unlabelled split pieces.

Prompt v2 on the 139 still-judged, hand-labelled items (a later review
round, 14 items, left these numbers essentially unchanged: 92% / 67% for
the chosen rule on all 145):

| rule for "real error" | precision | recall |
|---|---|---|
| DeepSeek@think (v1 prompt) | 65% | 92% |
| DeepSeek@think | 84% | 86% |
| MiMo@think | 92% | 54% |
| **DeepSeek@think, unless MiMo@think says either/Soniox wrong** | **93%** | **67%** |
| both say real error | 97% | 52% |

`@nothink` variants became much more conservative under v2 (recall
41–44%) and are dropped. DeepSeek@think fails to answer within 8k tokens
on about 11% of items (they stay Whisper's text). Cost: about $0.0016 per
item for DeepSeek@think and $0.00014 for MiMo@think.

Cheaper judges were also tried on the same 145 items (Ling 3.0 Flash, GLM
Flash, Mercury 2.5, Qwen3 235B, DeepSeek v4 Flash 0731). None matched
DeepSeek v4.1 as the main judge; DeepSeek 0731 is a third cheaper per call
but fails on 22% of items, dropping the rule's recall to 51–55%. Ling 3.0
is as good as MiMo as the cheap first pass, but MiMo is only ~13% of the cost.

### Training windows (`generate_targets.py`)

The edit model trains on windows of a row, not whole rows: every
difference in a training example has to be decided, and a whole row has
about 12, many in hard or garbled stretches. Windows let us pick the ones
that are cheap and clean to label.

```bash
python3 src/edit/generate_targets.py start --windows 2000   # background; resumable
python3 src/edit/generate_targets.py status                 # progress, cost so far + projected
python3 src/edit/generate_targets.py stop                   # graceful stop (in-flight calls saved)
python3 src/edit/generate_targets.py assemble               # rebuild train.jsonl from cached answers only
```

- **select**: windows never cut through a difference; skipped if they
  contain a garbled/long span. "Judged" windows have 1–3 substitution
  differences per 50 words (entity windows first); "free" windows have
  none, so they're no-edit examples at no cost. At most 2 windows per call.
  Insertions/deletions (words Whisper dropped or added) are out of scope for
  v1 and stay as Whisper has them.
- **judge**: MiMo first on every difference; DeepSeek only where MiMo
  didn't rule the edit out, in the same pool of parallel calls. Answers
  are cached as they arrive; `stop` or Ctrl+C saves calls in flight (up to
  60 s), and rerunning resumes without paying twice.
- **assemble**: an edit is accepted by the calibrated rule (hand labels
  override). A window with an *undecided* difference (DeepSeek ran out of
  tokens, a judge unsure, both versions wrong) is left out: it might hide a
  real error that the target would teach the model to leave in. Each window
  also stores 15 words of read-only context on each side
  (`left_context`/`right_context`); training uses plain windows by default,
  with overlapping windows at inference instead.

Two runs, on disjoint calls (`build_diffs.py --exclude-calls-from` built
the second pool from calls not used before):

| run | windows | with edits | edits | no-edit | excluded (undecided) | cost |
|---|---|---|---|---|---|---|
| `outputs/edit/data` (50 words) | 1,702 | 694 | 945 | 1,008 | 298 | $2.92 |
| `outputs/edit/data_v2` (40–80 words) | 1,328 | 597 | 910 | 731 | 232 | $3.07 |
| **total** | **3,030** | **1,291** | **1,855** | **1,739** | | **$5.99** |

2,879 train / 151 validation windows, split by call. About 8% of windows
are under 20 words (rows that are short themselves); the 109 under 10 words
add little and are meant to be filtered at training time.

## Step 3: the edit model (`train_edit.py`)

LoRA SFT of Qwen3.5-2B on the training windows. Format (`edit_format.py`):

```text
user:       Known names: نمونه‌پور، مریم
            [1]سلام [2]خانم [3]نمونپور [4]وقت [5]بخیر ...
assistant:  3 نمونپور → نمونه‌پور
            12-13 به سطح → بسته‌ت رو        (or NONE)
```

Word numbers are 1-based and ranges inclusive. Repeating the original words
(`data.target_includes_original`, on by default) costs a few tokens but lets
`parse_edits` reject a line whose numbers point at different words, instead
of corrupting the transcript. Lines that don't parse, fall outside the
window, overlap, or delete words are rejected and counted, never applied.

Inference on a whole transcript (`windowing.py`): `rules.py` first, then
50-word windows every 25 words; each word is owned by the one window where
it sits in the middle, so it's decided with at least 12 words of that window
on each side; an edit is kept only from its owner window. The evaluation
runs exactly this on the test set and scores the merged transcripts with
`scoring.py` next to two baselines (Whisper untouched, rules only), on the
main set and the entity/typo slices.

```bash
python3 src/edit/publish_edit_dataset.py                        # once: training windows -> private Hub dataset
python3 src/edit/train_edit.py --config configs/edit/smoke.yaml  # few steps + eval on 20 transcripts
python3 src/edit/train_edit.py --config configs/edit/qwen3.5-2b-edit.yaml
python3 src/run_sweep.py --sweep configs/edit/sweep.yaml --train_script src/edit/train_edit.py
```

Outputs in `output_dir`: `test_eval/` (`metrics.json` for the model, with
`wer` at the top for run_sweep.py; `baselines.json`; `comparison.md`;
`predictions.jsonl` with each transcript's edits and rejected lines),
`test_eval_entity/`, `test_eval_typo/`, and `validation_edits.json`
(exact-match edit precision/recall on the 144 held-out windows). Its config
schema is its own (`DEFAULTS` in `train_edit.py`, unknown keys rejected): the
rewrite pipeline's masking/weighting options assume a full-transcript
target. Entity metrics use Gemini labels only where `data/output-backup/`
exists (locally); elsewhere they fall back to CRM names.

### First run and threshold decoding

The first run (`edit-qwen3.5-2b` in the model repo): validation loss was
best at step 150 (end of epoch 1: 0.936), flat after, while training loss
halved -- the kept checkpoint is step 150. The model output NONE for almost
every window (41 edits in 12,934 test windows; 9 helped, 12 hurt), so the
result is essentially rules-only: WER 0.1863 vs Whisper 0.1929, 99.95% of
correct words kept (the rewrite run: WER 0.363, 76% kept). Likely causes:
greedy decoding puts all "no edit" probability on one token while edits are
spread over many word numbers; ~45% of training edits are `other_sub`
(sides don't sound alike, not inferable from text); few edits overall.

`decode_sweep.py` tests the first cause without retraining: per window it
measures p(NONE as first token), generates with NONE blocked, and uses the
blocked output where p_none < threshold; reports validation edit
precision/recall and full-pipeline test metrics per threshold, uploaded to
`<run-folder>/decode_sweep/`.

```bash
python3 src/edit/decode_sweep.py --run-folder edit-qwen3.5-2b
```

## Next steps

3. Run step 3 on a GPU and compare against `whisper`, `rules`, and the
   rewrite runs. If edits near window edges suffer, try the read-only
   context already stored with each window.
4. Add a separate span detector only if false edits remain a problem.
