"""Re-evaluate one checkpoint of a finished run against the main test set.

Fills a real gap in the pipeline. train.py runs the main test set exactly
twice -- once at baseline, once on the FINAL model -- and never on the best
checkpoint, because that one is only identified while training is still
going (see evaluate.py's update_best_checkpoint/TestEvalCallback). So a run
that early-stops, or whose quality peaks and then declines, ends up with a
test_eval/metrics.json describing a model nobody would ship.

That is not hypothetical: qwen3.5-2b-100pct-masked-weighted-v2 stopped at
step 4000, three checkpoints past its best at 2500, and its final-model WER
(0.2646) understated the best checkpoint's (0.2257) by four points.

Unlike evaluate.py's own --model_dir CLI, this reads the run's saved
resolved_config.yaml and reproduces train.py's final-eval call from it:
same system prompt, same generation settings, same corpus_freq (rebuilt
over the same train split, so fix_rate_recoverable's denominator matches
rather than silently differing). The numbers it writes are therefore
directly comparable to that run's own test_eval/ and test_eval_baseline/.
evaluate.py's CLI loads a whole model with AutoModelForCausalLM and has no
adapter path, so it cannot evaluate a LoRA checkpoint at all.

    python src/eval_checkpoint.py --run_dir outputs/qwen-mask-weighted/<run>

Writes <run_dir>/<output_subdir>/predictions.jsonl + metrics.json, defaulting
to test_eval_best/ so it never overwrites train.py's own test_eval/.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402
from data import load_sft_dataset, resolve_system_prompt  # noqa: E402
from evaluate import format_test_metrics, run_test_eval  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True,
                   help="A finished run's output_dir -- must hold resolved_config.yaml and the checkpoint below.")
    p.add_argument("--checkpoint", default="best_checkpoint_wer",
                   help="Adapter subdirectory inside --run_dir (default: best_checkpoint_wer; "
                        "also accepts e.g. checkpoint-2500).")
    p.add_argument("--output_subdir", default="test_eval_best",
                   help="Where to write predictions.jsonl + metrics.json, relative to --run_dir. "
                        "Deliberately NOT test_eval/, which train.py owns.")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Cap the test set (default: whatever the run's own config used).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(str(run_dir / "resolved_config.yaml"))
    adapter = run_dir / args.checkpoint
    if not adapter.exists():
        raise SystemExit(f"No such checkpoint: {adapter}")

    # Imported late so --help works without a GPU environment present.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"base={cfg.model_id}  adapter={adapter}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(run_dir))
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    # The same counter train.py builds, over the same split -- without it
    # fix_rate_recoverable is either None or computed against a different
    # denominator than the run's own numbers, and silently not comparable.
    print("building corpus_freq over the train split...", flush=True)
    raw = load_sft_dataset(
        cfg.dataset_id,
        cfg.train_split,
        cfg.eval_split,
        eval_fraction=cfg.eval_fraction,
        seed=cfg.seed,
        input_column=cfg.input_column,
        target_column=cfg.target_column,
        train_fraction=cfg.train_fraction,
    )
    corpus_freq = Counter()
    for text in raw["train"][cfg.target_column]:
        corpus_freq.update(text.split())
    print(f"corpus_freq: {len(corpus_freq)} distinct words", flush=True)

    metrics = run_test_eval(
        model,
        tokenizer,
        dataset_id=cfg.test_dataset_id,
        input_column=cfg.test_input_column,
        target_column=cfg.test_target_column,
        output_path=str(run_dir / args.output_subdir / "predictions.jsonl"),
        system_prompt=resolve_system_prompt(cfg),
        split=cfg.test_split,
        max_new_tokens=cfg.test_max_new_tokens,
        batch_size=cfg.test_batch_size,
        max_examples=args.max_examples if args.max_examples is not None else cfg.test_max_examples,
        hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
        premature_stop_ratio=cfg.test_premature_stop_ratio,
        premature_stop_min_input_words=cfg.test_premature_stop_min_input_words,
        repetition_penalty=cfg.test_repetition_penalty,
        no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
        fix_weight=cfg.test_fix_weight,
        low_signal_corpus_freq=corpus_freq,
        low_signal_min_similarity=cfg.mask_min_similarity,
        low_signal_max_common_freq=cfg.mask_max_common_freq,
    )
    print(f"{args.checkpoint}: {format_test_metrics(metrics)}", flush=True)


if __name__ == "__main__":
    main()
