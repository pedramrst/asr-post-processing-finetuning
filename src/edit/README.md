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

## Next steps

2. **Clean training targets.** Align Whisper vs Soniox, normalize with
   `persian_normalize`, drop what `rules.py` already fixes, then have cheap
   LLMs (via OpenRouter) classify each remaining difference as
   `real_error` / `style_variant` / `soniox_wrong` / `uncertain`, with CRM
   names as evidence. Keep an edit as a target only when the judges agree
   it's `real_error`. Validate the judges on ~150–200 hand-labelled
   differences first. Select a small, informative set (entity/CRM rows,
   phonetically close substitutions, plenty of no-edit rows) instead of the
   full corpus. The same judged labels also give a cleaner eval reference.
3. **Train the edit model** on word-indexed input (`[1]word [2]word …`)
   with a compact output (`3-4 → replacement`, or `NONE`), applied after
   `rules.py`. Compare against `whisper`, `rules`, and the rewrite runs.
4. Add a separate span detector only if false edits remain a problem.
