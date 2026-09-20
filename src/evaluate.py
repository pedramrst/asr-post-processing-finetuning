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
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import jiwer
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from transformers import TrainerCallback
from transformers.trainer import PREFIX_CHECKPOINT_DIR

from data import SYSTEM_PROMPT, load_local_jsonl_columns

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
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    """Greedy-decodes already chat-templated `prompts` in batches."""
    outputs = []
    prior_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # so every sequence in a batch ends at the same index
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prior_padding_side
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
) -> dict:
    ds = (
        load_local_jsonl_columns(dataset_id, [input_column, target_column])
        if Path(dataset_id).exists()
        else _load_hub_columns_pruned(dataset_id, [input_column, target_column], split)
    )
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
    predictions = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size)
    model.train(was_training)

    references = [ex[target_column] for ex in ds]
    corpus_wer = jiwer.process_words(references, predictions).wer if references else None
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

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, pred, ref, input_overlap, hallucinated in zip(
            ds, predictions, references, input_overlaps, hallucinated_flags
        ):
            row_wer = jiwer.process_words([ref], [pred]).wer if ref else None
            f.write(
                json.dumps(
                    {
                        "input": ex[input_column],
                        "output": pred,
                        "reference": ref,
                        "wer": row_wer,
                        "input_overlap_pct": input_overlap,
                        "hallucinated": hallucinated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    hallucination_rate = sum(hallucinated_flags) / len(hallucinated_flags) if hallucinated_flags else None
    metrics = {
        "wer": corpus_wer,
        "exact_match": exact_match,
        "hallucination_rate": hallucination_rate,
        "n_examples": len(ds),
    }
    out_path.with_name("metrics.json").write_text(json.dumps(metrics, indent=2))
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

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = None

    def on_save(self, args, state, control, **kwargs):
        metrics = run_test_eval(
            self.model,
            self.tokenizer,
            dataset_id=self.cfg.test_dataset_id,
            input_column=self.cfg.test_input_column,
            target_column=self.cfg.test_target_column,
            output_path=str(Path(self.cfg.output_dir) / "test_eval" / "predictions.jsonl"),
            system_prompt=self.cfg.system_prompt or SYSTEM_PROMPT,
            split=self.cfg.test_split,
            max_new_tokens=self.cfg.test_max_new_tokens,
            batch_size=self.cfg.test_batch_size,
            max_examples=self.cfg.test_max_examples,
            hallucination_overlap_floor=self.cfg.test_hallucination_overlap_floor,
        )

        checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        _, best_wer = update_best_checkpoint(self.cfg.output_dir, str(checkpoint_dir), metrics, state.global_step)

        if self.trainer is not None and metrics["wer"] is not None:
            log_values = {
                "test_wer": metrics["wer"],
                "test_exact_match": metrics["exact_match"],
                "test_hallucination_rate": metrics["hallucination_rate"],
            }
            if best_wer is not None:
                log_values["test_best_wer"] = best_wer
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
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _cli()
