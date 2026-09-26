"""Run the fine-tuned model on a test set, score WER, and save predictions.

Used both as a standalone CLI (to check any saved checkpoint by hand) and as
a Trainer callback (wired in by train.py) so results land in that run's
output_dir alongside its checkpoints, ready to be synced to the Hub by
hub_sync.py.

Can also be pointed at build_dataset.py's raw output or prepare_split.py's
curated one for a quick sanity check of this module itself, since both have
the same text_whisper/text columns as the real test set.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import jiwer
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from tqdm import tqdm
from transformers import TrainerCallback
from transformers.trainer import PREFIX_CHECKPOINT_DIR

from data import SYSTEM_PROMPT, load_local_jsonl_columns, low_signal_word_ranges, resolve_system_prompt
from persian_normalize import normalize_lenient, words_equivalent

load_dotenv()


def _load_hub_columns_pruned(repo_id: str, columns: list[str], split: str) -> Dataset:
    """Loads only `columns` from a Hub dataset's parquet shards, over HTTP.

    `datasets.load_dataset(repo_id, split=...)` downloads whole parquet
    files regardless of which columns you use afterwards -- fine for a
    text-only dataset, but callcc-test-1k (like callcc-2k, see
    build_dataset.py) has audio embedded in the same files. This mirrors
    build_dataset.py's technique: pyarrow's column selection over a
    random-access remote file object only fetches the byte ranges for the
    requested columns, so audio is never pulled over the wire.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{repo_id}/data/{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data/{split}-*.parquet files found in {repo_id}")
    tables = []
    for path in paths:
        with fs.open(path, "rb") as f:
            tables.append(pq.ParquetFile(f).read(columns=columns, use_threads=True))
    return Dataset(pa.concat_tables(tables))


def _equal_positions(alignment_chunks) -> set[int]:
    """Reference-word indices covered by an "equal" chunk of a jiwer word
    alignment -- i.e. words the hypothesis actually matched the reference
    on, at that exact position."""
    positions: set[int] = set()
    for chunk in alignment_chunks:
        if chunk.type == "equal":
            positions.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
    return positions


def _equal_or_equivalent_positions(alignment_chunks, ref_words: list[str], hyp_words: list[str]) -> set[int]:
    """Like _equal_positions, but a same-length "substitute" chunk also
    counts if every substituted word pair is persian_normalize.words_equivalent()
    -- informal/formal verb endings or را/رو attachment (see that module).
    Only ever evaluated on a pair the alignment already decided are a
    substitution for each other, which is what keeps it safe -- it can't
    fire on some unrelated word the way a blind text rewrite could. Used for
    the *_lenient metrics; _equal_positions (byte-exact only) is still what
    the raw/strict ones use."""
    positions: set[int] = set()
    for chunk in alignment_chunks:
        if chunk.type == "equal":
            positions.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
        elif chunk.type == "substitute":
            ref_span = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
            hyp_span = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
            if len(ref_span) == len(hyp_span) and all(
                words_equivalent(a, b) for a, b in zip(ref_span, hyp_span)
            ):
                positions.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
    return positions


def _strip_zwnj(text: str) -> str:
    """Removes ZWNJ (U+200C), the half-space Persian compound words use.

    "میگیره" (glued) and "می‌گیره" (correct ZWNJ half-space) become the same
    string here, so a WER computed on stripped text scores only genuine
    content/word-choice errors -- e.g. mis-heard names or terms -- separately
    from ZWNJ formatting, which the system prompt also asks for but is a much
    lower-stakes mistake than a wrong word.
    """
    return text.replace("‌", "")


def _word_overlap_pct(a: str, b: str) -> float:
    """% of `a`'s words that also appear (as a multiset) in `b`.

    Same metric as build_dataset.py's word_metrics(), applied here between a
    model's *output* and its *input* rather than text_whisper and the
    target -- a low score means the model produced words ungrounded in what
    it was actually given, i.e. hallucination, regardless of whether the
    output happens to match the reference.
    """
    wa = a.split()
    if not wa:
        return 100.0
    ca, cb = Counter(wa), Counter(b.split())
    return round(sum((ca & cb).values()) / len(wa) * 100, 2)


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> list[str]:
    """Greedy-decodes already chat-templated `prompts` in batches.

    Plain greedy decoding (the default `repetition_penalty=1.0`,
    `no_repeat_ngram_size=0` -- no-ops) is prone to a well-known degenerate
    failure: once a short, locally-high-probability phrase repeats a couple
    times (e.g. this task's "بله" agreement-word bursts, which do occur
    naturally as short runs in real training segments), greedy search can
    get stuck repeating it indefinitely instead of finding the actual
    stopping point, running all the way to max_new_tokens. Observed directly
    on this task's fine-tuned checkpoints: ~1 in 6-7 test outputs degenerated
    into a runaway repeat loop, tanking WER/hallucination_rate on otherwise
    good predictions. `repetition_penalty`/`no_repeat_ngram_size` are the
    standard mitigation.

    Batches are formed by length (longest first), not input order:
    `model.generate()` only stops a whole batch once every sequence in it
    has finished, so a single long prompt landing next to several short ones
    forces the short ones to keep decoding (and padding) far past where they
    would've stopped alone. This dataset's length spread is large (chunked
    rows ~15-40 words vs. assembled rows up to ~3,000), so an unsorted batch
    can be much slower than it needs to be. Sorting first (then restoring
    original order before returning) removes that waste for free; longest
    first also surfaces an OOM immediately rather than partway through.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    sorted_prompts = [prompts[i] for i in order]

    outputs_sorted = []
    prior_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # so every sequence in a batch ends at the same index
    try:
        batch_starts = range(0, len(sorted_prompts), batch_size)
        for i in tqdm(batch_starts, desc="generating", unit="batch"):
            batch = sorted_prompts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            outputs_sorted.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prior_padding_side

    outputs = [None] * len(prompts)
    for sorted_pos, original_idx in enumerate(order):
        outputs[original_idx] = outputs_sorted[sorted_pos]
    return outputs


def run_test_eval(
    model,
    tokenizer,
    dataset_id: str,
    input_column: str,
    target_column: str,
    output_path: str,
    system_prompt: str = SYSTEM_PROMPT,
    split: str = "test",
    max_new_tokens: int = 256,
    batch_size: int = 8,
    max_examples: int | None = None,
    hallucination_overlap_floor: float = 50.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    row_type_filter: str | None = None,
    fix_weight: float = 0.5,
    low_signal_corpus_freq: Counter | None = None,
    low_signal_min_similarity: float = 0.75,
    low_signal_max_common_freq: int = 1,
) -> dict:
    # row_type_filter is for a combined eval repo holding several eval
    # slices (e.g. entity + typo, see build_eval_dataset.py) distinguished
    # by a `row_type` column -- pull that column too, then narrow to just
    # this slice's rows, same as a dedicated single-slice dataset_id where
    # every row already qualifies.
    columns = [input_column, target_column] + (["row_type"] if row_type_filter else [])
    ds = (
        load_local_jsonl_columns(dataset_id, columns)
        if Path(dataset_id).exists()
        else _load_hub_columns_pruned(dataset_id, columns, split)
    )
    if row_type_filter:
        ds = ds.filter(lambda ex: ex["row_type"] == row_type_filter)
    # callcc-test-1k has a small fraction of rows (~1%) with a null
    # text_whisper -- filter before truncating to max_examples, so a small
    # slice doesn't end up mostly-nulls-that-get-dropped-anyway, and so a
    # None never reaches apply_chat_template as a message's content.
    ds = ds.filter(lambda ex: ex[input_column] and ex[target_column])
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": ex[input_column]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in ds
    ]

    was_training = model.training
    model.eval()
    predictions = generate_batch(
        model, tokenizer, prompts, max_new_tokens, batch_size,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
    )
    model.train(was_training)
    # Generation's KV cache grows token-by-token across variably-long
    # sequences, allocating/freeing many differently-sized tensors -- when
    # this runs mid-training (TestEvalCallback, in the same process as the
    # Trainer), that fragments PyTorch's CUDA caching allocator badly enough
    # that a later training step's backward pass can OOM trying to allocate
    # a single large contiguous block, even though nominal free memory looks
    # sufficient (observed: ~7.5GiB "reserved but unallocated" on a 32GB
    # card immediately breaking the next backward pass). Releasing eval's
    # cached blocks back to the driver here prevents that fragmentation from
    # carrying into training.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    references = [ex[target_column] for ex in ds]
    corpus_wer = jiwer.process_words(references, predictions).wer if references else None
    # Same WER, but with ZWNJ (half-space) differences removed from both
    # sides first -- isolates genuine content/word-choice errors (e.g.
    # mis-heard names) from spacing-only mismatches, which `corpus_wer` above
    # counts identically even though they're a much lower-stakes mistake.
    corpus_wer_zwnj_normalized = (
        jiwer.process_words(
            [_strip_zwnj(r) for r in references], [_strip_zwnj(p) for p in predictions]
        ).wer
        if references
        else None
    )
    # Stricter complement to WER (which gives partial credit for near
    # misses): what fraction did the model get exactly right. Also a direct
    # read on over/under-correction -- see build_example's system prompt
    # design and prepare_split.py's agree-bucket upsampling, both aimed at
    # this exact failure mode.
    exact_match = (
        sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(references)
        if references
        else None
    )

    # Flags predictions ungrounded in what the model was actually given --
    # distinct from wer/exact_match, which only compare against the
    # reference and can't tell "wrong correction" from "invented content".
    # `_word_overlap_pct(pred, input)` is the fraction of the *output*'s
    # words found in the *input*; a low score means the model said things
    # the input never did, regardless of whether it happens to match the
    # reference.
    input_overlaps = [_word_overlap_pct(pred, ex[input_column]) for ex, pred in zip(ds, predictions)]
    hallucinated_flags = [ov < hallucination_overlap_floor for ov in input_overlaps]

    # Targeted correction accuracy: WER/exact_match score the whole
    # sentence at once, which can't tell "fixed the actual errors" apart
    # from "left everything alone and got lucky" or "rewrote words that
    # were already right". This instead aligns text_whisper against the
    # reference (jiwer's word-level alignment, same tool as corpus_wer
    # above) to split every reference word into two buckets -- "target"
    # (Whisper got this wrong) and "already correct" (Whisper got this
    # right) -- then checks, per bucket, whether the prediction matches the
    # reference there. fix_rate is the first bucket's hit rate (did the
    # model actually correct what needed correcting); preservation_rate is
    # the second's (did the model leave what was already fine alone,
    # instead of introducing a new error). fix_weight (default 0.5, equal
    # weight -- see Config.test_fix_weight) combines them into one number;
    # both still get reported separately, since they're catching two
    # different failure modes and a single blended score can hide which one
    # is actually driving a change.
    fix_hits = fix_total = 0
    preserve_hits = preserve_total = 0

    # Lenient counterparts of the two above: same target/already-correct
    # split, but computed after persian_normalize.normalize_lenient() (ZWNJ/
    # spacing + a curated informal-word dictionary) and with a same-length
    # substitution counted as a match when persian_normalize.words_equivalent()
    # says so (informal/formal verb endings, را/رو attachment) -- see that
    # module for exactly what does and doesn't get normalized, and why (it
    # was scoped down after finding real cases where the obvious approach
    # mis-"corrected" actual entity names/brands).
    fix_hits_lenient = fix_total_lenient = 0
    preserve_hits_lenient = preserve_total_lenient = 0

    # fix_rate_lenient still counts corrections Whisper left no recoverable
    # signal for -- a word it garbled beyond recognition, or never produced
    # at all -- which no model can fix except by guessing, so they cap the
    # metric below 1.0 no matter how good the model is. Dropping them from
    # the denominator makes it "did it fix what was actually fixable"
    # instead. Detection is data.py's low_signal_word_ranges(), the same one
    # Config.mask_low_signal_corrections uses to drop these from the
    # *training* loss -- so what training declines to teach and what eval
    # declines to score stay in sync by construction. Needs a corpus
    # frequency counter to tell an unrecoverable rare name from a merely
    # dropped "بله" (trivially recoverable from context regardless of what
    # Whisper said); without one this stays None rather than being computed
    # against a different-meaning denominator -- see the low_signal_corpus_freq
    # argument.
    fix_hits_recoverable = fix_total_recoverable = 0
    unrecoverable_targets = 0

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, pred, ref, input_overlap, hallucinated in zip(
            ds, predictions, references, input_overlaps, hallucinated_flags
        ):
            pred_alignment = jiwer.process_words([ref], [pred]) if ref else None
            row_wer = pred_alignment.wer if pred_alignment else None
            row_wer_zwnj_normalized = (
                jiwer.process_words([_strip_zwnj(ref)], [_strip_zwnj(pred)]).wer if ref else None
            )

            row_fix_rate = row_preservation_rate = None
            row_fix_rate_lenient = row_preservation_rate_lenient = None
            row_fix_rate_recoverable = None
            if ref and ex[input_column] and pred_alignment:
                whisper_alignment = jiwer.process_words([ref], [ex[input_column]])
                already_correct = _equal_positions(whisper_alignment.alignments[0])
                target = set(range(len(ref.split()))) - already_correct
                still_correct = _equal_positions(pred_alignment.alignments[0])

                row_fix_hits = len(target & still_correct)
                row_preserve_hits = len(already_correct & still_correct)
                fix_hits += row_fix_hits
                fix_total += len(target)
                preserve_hits += row_preserve_hits
                preserve_total += len(already_correct)

                row_fix_rate = row_fix_hits / len(target) if target else None
                row_preservation_rate = row_preserve_hits / len(already_correct) if already_correct else None

                ref_l = normalize_lenient(ref)
                whisper_l = normalize_lenient(ex[input_column])
                pred_l = normalize_lenient(pred)
                ref_l_words, whisper_l_words, pred_l_words = ref_l.split(), whisper_l.split(), pred_l.split()
                whisper_l_alignment = jiwer.process_words([ref_l], [whisper_l])
                pred_l_alignment = jiwer.process_words([ref_l], [pred_l])

                already_correct_l = _equal_or_equivalent_positions(
                    whisper_l_alignment.alignments[0], ref_l_words, whisper_l_words)
                target_l = set(range(len(ref_l_words))) - already_correct_l
                still_correct_l = _equal_or_equivalent_positions(
                    pred_l_alignment.alignments[0], ref_l_words, pred_l_words)

                row_fix_hits_l = len(target_l & still_correct_l)
                row_preserve_hits_l = len(already_correct_l & still_correct_l)
                fix_hits_lenient += row_fix_hits_l
                fix_total_lenient += len(target_l)
                preserve_hits_lenient += row_preserve_hits_l
                preserve_total_lenient += len(already_correct_l)

                row_fix_rate_lenient = row_fix_hits_l / len(target_l) if target_l else None
                row_preservation_rate_lenient = (
                    row_preserve_hits_l / len(already_correct_l) if already_correct_l else None)

                if low_signal_corpus_freq is not None:
                    unrecoverable = set()
                    for first, last in low_signal_word_ranges(
                        whisper_l, ref_l, low_signal_min_similarity,
                        low_signal_corpus_freq, low_signal_max_common_freq,
                    ):
                        unrecoverable.update(range(first, last + 1))
                    target_r = target_l - unrecoverable
                    row_fix_hits_r = len(target_r & still_correct_l)
                    fix_hits_recoverable += row_fix_hits_r
                    fix_total_recoverable += len(target_r)
                    unrecoverable_targets += len(target_l & unrecoverable)
                    row_fix_rate_recoverable = row_fix_hits_r / len(target_r) if target_r else None

            f.write(
                json.dumps(
                    {
                        "input": ex[input_column],
                        "output": pred,
                        "reference": ref,
                        "wer": row_wer,
                        "wer_zwnj_normalized": row_wer_zwnj_normalized,
                        "input_overlap_pct": input_overlap,
                        "hallucinated": hallucinated,
                        "fix_rate": row_fix_rate,
                        "preservation_rate": row_preservation_rate,
                        "fix_rate_lenient": row_fix_rate_lenient,
                        "preservation_rate_lenient": row_preservation_rate_lenient,
                        "fix_rate_recoverable": row_fix_rate_recoverable,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    hallucination_rate = sum(hallucinated_flags) / len(hallucinated_flags) if hallucinated_flags else None
    fix_rate = fix_hits / fix_total if fix_total else None
    preservation_rate = preserve_hits / preserve_total if preserve_total else None
    targeted_score = (
        fix_weight * fix_rate + (1 - fix_weight) * preservation_rate
        if fix_rate is not None and preservation_rate is not None
        else None
    )
    fix_rate_lenient = fix_hits_lenient / fix_total_lenient if fix_total_lenient else None
    preservation_rate_lenient = preserve_hits_lenient / preserve_total_lenient if preserve_total_lenient else None
    targeted_score_lenient = (
        fix_weight * fix_rate_lenient + (1 - fix_weight) * preservation_rate_lenient
        if fix_rate_lenient is not None and preservation_rate_lenient is not None
        else None
    )
    fix_rate_recoverable = (
        fix_hits_recoverable / fix_total_recoverable
        if low_signal_corpus_freq is not None and fix_total_recoverable
        else None
    )
    metrics = {
        "wer": corpus_wer,
        "wer_zwnj_normalized": corpus_wer_zwnj_normalized,
        "exact_match": exact_match,
        "fix_rate": fix_rate,
        "preservation_rate": preservation_rate,
        "targeted_score": targeted_score,
        "fix_rate_lenient": fix_rate_lenient,
        "preservation_rate_lenient": preservation_rate_lenient,
        "targeted_score_lenient": targeted_score_lenient,
        "fix_rate_recoverable": fix_rate_recoverable,
        "unrecoverable_targets": unrecoverable_targets if low_signal_corpus_freq is not None else None,
        "hallucination_rate": hallucination_rate,
        "n_examples": len(ds),
    }
    # Atomic write (temp file + os.replace), not a direct write_text(): a
    # plain open("w") truncates the file before writing its content, so an
    # external reader (e.g. the Telegram agent's tools.py polling this file)
    # can race and read "" mid-write. Writing to a temp file in the same
    # directory first and replacing it makes the read atomic for any reader.
    metrics_path = out_path.with_name("metrics.json")
    fd, tmp_name = tempfile.mkstemp(dir=metrics_path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    os.replace(tmp_name, metrics_path)
    return metrics


def update_best_checkpoint(output_dir: str, source_dir: str, metrics: dict, step: int) -> tuple[bool, float | None]:
    """Copies source_dir into <output_dir>/best_checkpoint_wer/ if metrics['wer']
    beats whatever's recorded there -- read back from that folder's own
    best_metrics.json rather than kept in memory, so this is correct across
    resumed runs too, not just within one process's lifetime.

    source_dir may be a numbered checkpoint dir, or output_dir itself (the
    post-training final-model case in train.py) -- the ignore patterns below
    exist specifically so the latter doesn't try to copy output_dir into a
    subdirectory of itself.

    Returns (updated, current_best_wer) -- current_best_wer is the running
    best after this call, whether or not this call was the one that set it,
    so callers can log a monotonic "best so far" curve either way.
    """
    if metrics.get("wer") is None:
        return False, None

    best_dir = Path(output_dir) / "best_checkpoint_wer"
    best_metrics_path = best_dir / "best_metrics.json"
    if best_metrics_path.exists():
        current_best = json.loads(best_metrics_path.read_text())["wer"]
        if metrics["wer"] >= current_best:
            return False, current_best

    tmp_dir = tempfile.mkdtemp(prefix="best_checkpoint_wer_")
    shutil.copytree(
        source_dir,
        tmp_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("best_checkpoint_wer", f"{PREFIX_CHECKPOINT_DIR}-*", "test_eval", "tb"),
    )
    (Path(tmp_dir) / "best_metrics.json").write_text(json.dumps({"step": step, **metrics}, indent=2))
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.move(tmp_dir, str(best_dir))
    return True, metrics["wer"]


def run_secondary_eval(model, tokenizer, cfg, dataset_id: str | None, name: str, output_subdir: str,
                        trainer=None, low_signal_corpus_freq: Counter | None = None):
    """Runs run_test_eval() against one of cfg's secondary datasets (e.g.
    test_entity_dataset_id, test_typo_dataset_id) -- a no-op returning None
    if `dataset_id` is unset. Shared by TestEvalCallback and train.py's
    baseline/final eval blocks so adding another named slice (beyond entity
    and typo) doesn't mean copy-pasting a fourth near-identical block.

    Writes to <output_dir>/<output_subdir>/, reusing every other test.*
    generation setting from cfg. If `trainer` is given, logs
    test_<name>_wer/wer_zwnj_normalized/exact_match/hallucination_rate/
    fix_rate/preservation_rate/targeted_score.

    `name` also selects cfg.test_<name>_row_type (e.g. test_entity_row_type)
    -- set that when dataset_id points at a combined eval repo holding
    several slices distinguished by a `row_type` column, so this only scores
    the rows belonging to this slice. Leave it unset (the default) for a
    dedicated per-slice repo/local file, where every row already qualifies.
    """
    if not dataset_id:
        return None
    metrics = run_test_eval(
        model,
        tokenizer,
        dataset_id=dataset_id,
        input_column=cfg.test_input_column,
        target_column=cfg.test_target_column,
        output_path=str(Path(cfg.output_dir) / output_subdir / "predictions.jsonl"),
        system_prompt=resolve_system_prompt(cfg),
        split=cfg.test_split,
        max_new_tokens=cfg.test_max_new_tokens,
        batch_size=cfg.test_batch_size,
        row_type_filter=getattr(cfg, f"test_{name}_row_type", None),
        hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
        repetition_penalty=cfg.test_repetition_penalty,
        no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
        fix_weight=cfg.test_fix_weight,
        low_signal_corpus_freq=low_signal_corpus_freq,
        low_signal_min_similarity=cfg.mask_min_similarity,
        low_signal_max_common_freq=cfg.mask_max_common_freq,
    )
    if trainer is not None and metrics["wer"] is not None:
        trainer.log({
            f"test_{name}_wer": metrics["wer"],
            f"test_{name}_wer_zwnj_normalized": metrics["wer_zwnj_normalized"],
            f"test_{name}_exact_match": metrics["exact_match"],
            f"test_{name}_hallucination_rate": metrics["hallucination_rate"],
            f"test_{name}_fix_rate": metrics["fix_rate"],
            f"test_{name}_preservation_rate": metrics["preservation_rate"],
            f"test_{name}_targeted_score": metrics["targeted_score"],
            f"test_{name}_fix_rate_lenient": metrics["fix_rate_lenient"],
            f"test_{name}_preservation_rate_lenient": metrics["preservation_rate_lenient"],
            f"test_{name}_targeted_score_lenient": metrics["targeted_score_lenient"],
            **({f"test_{name}_fix_rate_recoverable": metrics["fix_rate_recoverable"]}
                if metrics["fix_rate_recoverable"] is not None else {}),
        })
    return metrics


class TestEvalCallback(TrainerCallback):
    """Runs run_test_eval() on every checkpoint save; logs WER to TensorBoard.

    Must subclass TrainerCallback (not just define on_save) -- the Trainer's
    callback handler fires many other events too (on_init_end, on_log, ...)
    and calls getattr(callback, event) unconditionally, which would raise
    AttributeError on the first non-on_save event without the base class's
    no-op stubs for everything else.

    `trainer` is set after Trainer construction (train.py) so this can call
    trainer.log() -- the Trainer isn't available yet when callbacks are
    built, since callbacks are passed *into* the Trainer constructor.
    """

    def __init__(self, cfg, model, tokenizer, low_signal_corpus_freq: Counter | None = None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = None
        self.low_signal_corpus_freq = low_signal_corpus_freq

    def on_save(self, args, state, control, **kwargs):
        # This fires on every checkpoint save (potentially hundreds of times
        # over a long run). Entity/typo run first and unconditionally -- if
        # test_checkpoint_eval_main is off, best-checkpoint tracking below
        # falls back to these instead of going inert for the whole run.
        entity_metrics = run_secondary_eval(
            self.model, self.tokenizer, self.cfg, self.cfg.test_entity_dataset_id,
            "entity", "test_eval_entity", self.trainer, self.low_signal_corpus_freq,
        )
        typo_metrics = run_secondary_eval(
            self.model, self.tokenizer, self.cfg, self.cfg.test_typo_dataset_id,
            "typo", "test_eval_typo", self.trainer, self.low_signal_corpus_freq,
        )

        # test_checkpoint_eval_main (default True) gates only this per-
        # checkpoint pass -- baseline/final main-test evals in train.py
        # always run regardless. test_checkpoint_max_examples, when set,
        # keeps this cheap; test_max_examples (the full set, typically) is
        # reserved for the one-time baseline/final evals in train.py.
        metrics = None
        log_values: dict = {}
        if self.cfg.test_checkpoint_eval_main:
            max_examples = (
                self.cfg.test_checkpoint_max_examples
                if self.cfg.test_checkpoint_max_examples is not None
                else self.cfg.test_max_examples
            )
            metrics = run_test_eval(
                self.model,
                self.tokenizer,
                dataset_id=self.cfg.test_dataset_id,
                input_column=self.cfg.test_input_column,
                target_column=self.cfg.test_target_column,
                output_path=str(Path(self.cfg.output_dir) / "test_eval" / "predictions.jsonl"),
                system_prompt=resolve_system_prompt(self.cfg),
                split=self.cfg.test_split,
                max_new_tokens=self.cfg.test_max_new_tokens,
                batch_size=self.cfg.test_batch_size,
                max_examples=max_examples,
                hallucination_overlap_floor=self.cfg.test_hallucination_overlap_floor,
                repetition_penalty=self.cfg.test_repetition_penalty,
                no_repeat_ngram_size=self.cfg.test_no_repeat_ngram_size,
                fix_weight=self.cfg.test_fix_weight,
                low_signal_corpus_freq=self.low_signal_corpus_freq,
                low_signal_min_similarity=self.cfg.mask_min_similarity,
                low_signal_max_common_freq=self.cfg.mask_max_common_freq,
            )
            if metrics["wer"] is not None:
                log_values = {
                    "test_wer": metrics["wer"],
                    "test_wer_zwnj_normalized": metrics["wer_zwnj_normalized"],
                    "test_exact_match": metrics["exact_match"],
                    "test_hallucination_rate": metrics["hallucination_rate"],
                    "test_fix_rate": metrics["fix_rate"],
                    "test_preservation_rate": metrics["preservation_rate"],
                    "test_targeted_score": metrics["targeted_score"],
                    "test_fix_rate_lenient": metrics["fix_rate_lenient"],
                    "test_preservation_rate_lenient": metrics["preservation_rate_lenient"],
                    "test_targeted_score_lenient": metrics["targeted_score_lenient"],
                }
                if metrics["fix_rate_recoverable"] is not None:
                    log_values["test_fix_rate_recoverable"] = metrics["fix_rate_recoverable"]

        # Best-checkpoint tracking (see Config.test_checkpoint_eval_main's
        # docstring for the fallback's reasoning): prefer the main test
        # set's WER; when that didn't run this checkpoint, fall back to the
        # mean of entity/typo WER instead of going inert.
        best_source = metrics
        if best_source is None:
            wers = [m["wer"] for m in (entity_metrics, typo_metrics) if m is not None and m.get("wer") is not None]
            best_source = {"wer": sum(wers) / len(wers) if wers else None}

        checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        _, best_wer = update_best_checkpoint(self.cfg.output_dir, str(checkpoint_dir), best_source, state.global_step)
        if best_wer is not None:
            log_values["test_best_wer"] = best_wer

        if self.trainer is not None and log_values:
            self.trainer.log(log_values)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True, help="Local path (or Hub repo id) with the base model + LoRA adapter.")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--input_column", default="text_whisper")
    p.add_argument("--target_column", default="text")
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--hallucination_overlap_floor", type=float, default=50.0,
                    help="Flag a prediction as hallucinated when under this %% of its words appear in the input.")
    p.add_argument("--repetition_penalty", type=float, default=1.2,
                    help="Penalizes repeated tokens during greedy decoding -- mitigates runaway repeat loops "
                         "(e.g. this task's short natural agreement-word bursts spiraling into hundreds of "
                         "repeats). 1.0 disables it.")
    p.add_argument("--no_repeat_ngram_size", type=int, default=3,
                    help="Hard-blocks repeating any n-gram of this size that's already appeared. 0 disables it.")
    args = p.parse_args()

    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=_torch.bfloat16, device_map="auto")

    metrics = run_test_eval(
        model,
        tokenizer,
        dataset_id=args.dataset_id,
        input_column=args.input_column,
        target_column=args.target_column,
        output_path=args.output,
        system_prompt=args.system_prompt,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        hallucination_overlap_floor=args.hallucination_overlap_floor,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _cli()
