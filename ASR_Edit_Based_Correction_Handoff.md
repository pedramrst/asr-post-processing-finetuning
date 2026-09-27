# ASR correction by detecting and applying minimal edits

## Context and objective

The ASR transcript is typically 100–200 words long, with only a few mistakes. Errors include ordinary recognition mistakes and misrecognized brand or product names. Fine-tuning an LLM to rewrite the entire transcript has caused unwanted changes to words that were already correct. The objective is to improve entity accuracy while preserving every unaffected part of the original transcript.

## Proposed design

Train a model to return **only proposed edits**, not a rewritten transcript. Each edit identifies a span in the original ASR output and a replacement. Application code validates and applies edits to the original string; an empty edit list leaves the transcript unchanged.

Example:

```text
ASR: I ordered the Sam sung galaxy from Tech Mart yesterday.
Edits: [{"original":"Sam sung galaxy","replacement":"Samsung Galaxy"}]
Result: I ordered the Samsung Galaxy from Tech Mart yesterday.
```

Start with a single small instruction model that jointly locates and corrects errors. Its output should be a constrained JSON edit list. For production, use offsets into the **original input string** (define whether these are Unicode code point or byte offsets) and include the original span for validation. Require nonoverlapping spans and apply replacements from right to left. Reject malformed edits and edits whose `original` value does not exactly match the referenced substring. Preserve whitespace and punctuation outside edited spans.

Example target schema:

```json
{"edits":[{"start":14,"end":29,"original":"Sam sung galaxy","replacement":"Samsung Galaxy","type":"product"}]}
```

The offsets above are illustrative; compute actual offsets from the input during dataset creation rather than copying the example. It is also reasonable to initially train on exact quoted spans and add offsets once alignment is reliable. Duplicate occurrences require offsets or another unambiguous span identifier.

## Entity grounding

Retrieve a short list of plausible brand and product names from the current catalog, using spelling variants, aliases, phonetic similarity, and relevant context. Pass candidates to the corrector. Keep the catalog separate from model weights so newly introduced products can be handled without retraining. Retrieval candidates are evidence, not an instruction to force a replacement. If the ASR exposes alternative hypotheses, word timestamps, confidence, or audio, use them when available; text alone may not disambiguate similar sounding entities.

## Training data

- Collect representative pairs of raw ASR output and manually verified reference transcripts from the actual domain and languages. Preserve raw ASR text before any normalization.
- Align each pair into minimal replacement, insertion, and deletion edits. Review ambiguous alignments and entity boundaries; a plain text diff can be misleading when words repeat or spacing differs.
- Include many clean transcripts with `{"edits":[]}` and examples containing correct but unusual brand names. Include near misses and confusable catalog entries.
- Include actual ASR errors and optionally synthetic perturbations that resemble the ASR system's mistakes, especially phonetic substitutions, word splits/merges, and code switching if relevant. Do not rely solely on synthetic typos.
- Make a time- or entity-disjoint test split for new products. Keep related utterances and speakers from leaking across training and test where possible.

## If the single model over-edits

Use a two-stage system: (1) an encoder-based token/span detector flags suspicious regions; (2) a small LLM or encoder–decoder model sees each flagged region, its surrounding context, and retrieved entity candidates, and returns a replacement or `KEEP`. The final text is still assembled by deterministic code. This permits separate thresholds for detection and correction, at the cost of a more complex pipeline and potential missed errors at stage one. A fine-tuned mT5 can be evaluated as a **span corrector**; it need not generate all 200 words.

## Evaluation and decision criteria

Compare at least three baselines: untouched ASR, full-transcript rewriting, and the edit-list system. If useful, compare the two-stage detector/corrector. Report:

- Word error rate or character error rate as appropriate to the language and normalization policy.
- Entity recall and precision, plus exact-match accuracy on brand/product names, including unseen products.
- **False edits on clean text**: proportion of initially correct transcripts changed, and incorrect edits per 100 originally correct words.
- Edit detection precision/recall separately from correction accuracy conditional on detecting the right span.
- Latency and cost per transcript; invalid edit rate and abstention rate.

Tune the acceptance threshold for the application: if changing a correct product is expensive, favor high precision and leave uncertain spans unchanged or route them for review. Evaluate whole-transcript outcomes, because a successful entity correction does not compensate for damage elsewhere.

## Suggested next implementation steps

1. Define the precise correction scope: entities only, or all ASR errors; languages; acceptable punctuation/casing edits; access to audio/ASR alternatives.
2. Build a small, audited evaluation set with clean transcripts, entity mistakes, and unseen catalog items.
3. Implement pair-to-edit alignment, an edit validator/applicator, and retrieval from the product catalog.
4. Fine-tune a small model on edit-list targets and compare against the baselines using the same test set.
5. Add a separate span detector only if false edits remain a material issue.

## Open questions for the next agent

- What languages and scripts appear in transcripts? Are there mixed-language brand names?
- How many verified ASR/reference pairs and clean transcripts are available?
- Is there an authoritative, regularly updated catalog with aliases and pronunciations?
- Are audio, ASR alternatives, confidence scores, and timestamps available?
- Is the primary constraint false corrections, missed corrections, latency, or compute?

## Relevant starting points

- GECToR, *Grammatical Error Correction: Tag, Not Rewrite*: https://aclanthology.org/2020.bea-1.16/
- *Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction*: https://aclanthology.org/2020.emnlp-main.581/
- *Retrieval-Augmented Correction of Named Entity Speech Recognition Errors*: https://machinelearning.apple.com/research/retrieval-asr
- *ASR Error Correction with Augmented Transformer for Entity Retrieval*: https://www.isca-archive.org/interspeech_2020/wang20p_interspeech.html
