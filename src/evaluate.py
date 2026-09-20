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
from pathlib import Path

import jiwer
import torch
from datasets import load_dataset
from transformers import TrainerCallback

from data import SYSTEM_PROMPT


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
) -> dict:
    ds = (
        load_dataset("json", data_files=dataset_id)["train"]
        if Path(dataset_id).exists()
        else load_dataset(dataset_id, split=split)
    )
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

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, pred, ref in zip(ds, predictions, references):
            row_wer = jiwer.process_words([ref], [pred]).wer if ref else None
            f.write(
                json.dumps(
                    {"input": ex[input_column], "output": pred, "reference": ref, "wer": row_wer},
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {"wer": corpus_wer, "n_examples": len(ds)}
    out_path.with_name("metrics.json").write_text(json.dumps(metrics, indent=2))
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
        )
        if self.trainer is not None and metrics["wer"] is not None:
            self.trainer.log({"test_wer": metrics["wer"]})


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
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _cli()
