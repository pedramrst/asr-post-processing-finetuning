"""LoRA SFT for correcting Persian Whisper ASR output against frontier-model transcripts.

Data is expected to already be preprocessed by the data team and published as a
Hugging Face dataset with an input column (default: `text_whisper`, the in-house
Whisper output) and a target column (default: `text`, the frontier-model
transcript). This script only handles tokenization, LoRA setup, and training —
see data.py:load_sft_dataset for the (thin) loading step to swap in your
teammate's actual loading/post-processing call.

All hyperparameters live in a YAML config (see configs/base.yaml) rather than
CLI flags, so a run is fully reproducible from one file. Use --set for quick
one-off overrides without editing the file, e.g.:

    python train.py --config configs/base.yaml --set training.learning_rate=1e-4 --set lora.r=32

To queue up several runs back-to-back for comparison, use run_sweep.py instead
of calling this script directly.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from config import Config, load_config
from data import SYSTEM_PROMPT, PadCollator, build_example, load_sft_dataset
from evaluate import TestEvalCallback, run_test_eval, update_best_checkpoint
from hub_sync import SyncToHubCallback, repo_folder_name, sync_output_dir

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a YAML config file (see configs/base.yaml).")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="section.key=value",
        help="Override a single config value, e.g. --set training.learning_rate=1e-4. Repeatable.",
    )
    return p.parse_args()


def _parse_overrides(raw_overrides: list[str]) -> dict:
    overrides: dict = {}
    for item in raw_overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"--set expects section.key=value, got: {item!r}")
        parsed_value = yaml.safe_load(value)
        if "." in key:
            section, sub_key = key.split(".", 1)
            overrides.setdefault(section, {})[sub_key] = parsed_value
        else:
            overrides[key] = parsed_value
    return overrides


def resolve_resume_checkpoint(cfg: Config) -> bool | str | None:
    resume = cfg.resume_from_checkpoint
    if resume is False:
        return None
    if resume is True:
        return True
    if Path(resume).exists():
        return resume
    if resume != "hub":
        raise ValueError(f"resume_from_checkpoint={resume!r} is neither an existing local path nor 'hub'.")

    from huggingface_hub import snapshot_download

    folder = repo_folder_name(cfg)
    print(f"Downloading '{folder}' from {cfg.hub_repo_id} into {cfg.output_dir} ...")
    # local_dir is output_dir's *parent* since snapshot_download recreates the
    # repo-relative path (<folder>/...) under it -- this only lands exactly in
    # cfg.output_dir when hub_repo_folder matches output_dir's basename (the
    # default), which is the assumption this resume path relies on.
    snapshot_download(
        repo_id=cfg.hub_repo_id,
        allow_patterns=[f"{folder}/**"],
        local_dir=str(Path(cfg.output_dir).parent),
    )
    return True


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=_parse_overrides(args.overrides))

    if cfg.push_to_hub and not cfg.hub_repo_id:
        raise ValueError("hub.push_to_hub is true but hub.repo_id is not set.")
    if cfg.on_long_example not in ("drop", "truncate"):
        raise ValueError(f"on_long_example must be 'drop' or 'truncate', got {cfg.on_long_example!r}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Saved alongside checkpoints/tb logs so this run is self-describing --
    # the load_model notebook reads model_id/system_prompt back out of this.
    (output_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    # Deterministic path (not transformers' default output_dir/runs/<timestamp>)
    # so every run's TensorBoard logs land in the same place inside output_dir.
    tb_dir = cfg.tensorboard_logging_dir or str(output_dir / "tb")
    os.environ["TENSORBOARD_LOGGING_DIR"] = tb_dir

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
    if cfg.train_fraction is not None and cfg.train_fraction < 1.0:
        print(f"train_fraction={cfg.train_fraction}: using {len(raw['train'])} train rows")

    length_stats = Counter()

    def _map(example):
        kwargs = {} if cfg.system_prompt is None else {"system_prompt": cfg.system_prompt}
        result = build_example(
            tokenizer,
            example[cfg.input_column],
            example[cfg.target_column],
            max_length=cfg.max_length,
            on_long_example=cfg.on_long_example,
            **kwargs,
        )
        length_stats[result["status"]] += 1
        return result

    tokenized = raw.map(_map, remove_columns=raw["train"].column_names)

    n_total = sum(length_stats.values())
    print(
        f"Tokenization ({n_total} rows, train+eval combined): "
        f"{length_stats['ok']} ok, {length_stats['truncated']} truncated, "
        f"{length_stats['dropped']} dropped (exceeded max_length={cfg.max_length})"
    )
    if length_stats["dropped"]:
        tokenized = tokenized.filter(lambda ex: ex["status"] != "dropped")
    tokenized = tokenized.remove_columns("status")

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    has_eval = "validation" in tokenized
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        # `warmup_ratio` was folded into `warmup_steps` (float < 1 == ratio) in this
        # transformers version; kept as `warmup_ratio` in our own config for clarity.
        warmup_steps=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg.eval_steps if has_eval else None,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        disable_tqdm=False,
        seed=cfg.seed,
        # Hub sync is handled ourselves (hub_sync.py), not Trainer's built-in
        # push_to_hub/hub_strategy -- those assume the whole target repo IS
        # this one run, which doesn't fit "one shared repo, folder per run".
    )

    callbacks = []
    test_eval_callback = None
    if cfg.test_dataset_id:
        test_eval_callback = TestEvalCallback(cfg, model, tokenizer)
        callbacks.append(test_eval_callback)
    if cfg.push_to_hub:
        # Registered after the test-eval callback: callbacks fire in list
        # order, so each save's freshly written test_eval/ files are already
        # on disk by the time this uploads output_dir.
        callbacks.append(SyncToHubCallback(cfg))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
        callbacks=callbacks,
    )
    if test_eval_callback is not None:
        test_eval_callback.trainer = trainer

    resume_checkpoint = resolve_resume_checkpoint(cfg)

    # LoRA's B matrix is zero-initialized, so the just-wrapped, untrained
    # model is numerically identical to the plain base model right now --
    # this doubles as a pre-fine-tuning baseline without a separate
    # unadapted model load. Skipped on a resumed run, since step 0 there
    # already has a real baseline from the original run.
    if cfg.test_dataset_id and cfg.test_baseline and resume_checkpoint is None:
        print("Baseline test-set eval (before any training steps)...")
        baseline_metrics = run_test_eval(
            model,
            tokenizer,
            dataset_id=cfg.test_dataset_id,
            input_column=cfg.test_input_column,
            target_column=cfg.test_target_column,
            output_path=str(output_dir / "test_eval_baseline" / "predictions.jsonl"),
            system_prompt=cfg.system_prompt or SYSTEM_PROMPT,
            split=cfg.test_split,
            max_new_tokens=cfg.test_max_new_tokens,
            batch_size=cfg.test_batch_size,
            max_examples=cfg.test_max_examples,
            hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
            repetition_penalty=cfg.test_repetition_penalty,
            no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
        )
        print(f"Baseline: wer={baseline_metrics['wer']}, exact_match={baseline_metrics['exact_match']}, "
              f"hallucination_rate={baseline_metrics['hallucination_rate']}")
        if baseline_metrics["wer"] is not None:
            trainer.log({
                "test_wer": baseline_metrics["wer"],
                "test_wer_zwnj_normalized": baseline_metrics["wer_zwnj_normalized"],
                "test_exact_match": baseline_metrics["exact_match"],
                "test_hallucination_rate": baseline_metrics["hallucination_rate"],
            })

        if cfg.test_entity_dataset_id:
            baseline_entity_metrics = run_test_eval(
                model,
                tokenizer,
                dataset_id=cfg.test_entity_dataset_id,
                input_column=cfg.test_input_column,
                target_column=cfg.test_target_column,
                output_path=str(output_dir / "test_eval_entity_baseline" / "predictions.jsonl"),
                system_prompt=cfg.system_prompt or SYSTEM_PROMPT,
                split=cfg.test_split,
                max_new_tokens=cfg.test_max_new_tokens,
                batch_size=cfg.test_batch_size,
                hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
                repetition_penalty=cfg.test_repetition_penalty,
                no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
            )
            print(f"Baseline (entity slice): wer={baseline_entity_metrics['wer']}, "
                  f"exact_match={baseline_entity_metrics['exact_match']}")
            if baseline_entity_metrics["wer"] is not None:
                trainer.log({
                    "test_entity_wer": baseline_entity_metrics["wer"],
                    "test_entity_wer_zwnj_normalized": baseline_entity_metrics["wer_zwnj_normalized"],
                    "test_entity_exact_match": baseline_entity_metrics["exact_match"],
                    "test_entity_hallucination_rate": baseline_entity_metrics["hallucination_rate"],
                })

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # The final adapter/tokenizer save above happens after train() returns,
    # so it's not covered by the last on_save-triggered callback run -- redo
    # test eval + hub sync once more here to capture it.
    if cfg.test_dataset_id:
        final_metrics = run_test_eval(
            model,
            tokenizer,
            dataset_id=cfg.test_dataset_id,
            input_column=cfg.test_input_column,
            target_column=cfg.test_target_column,
            output_path=str(output_dir / "test_eval" / "predictions.jsonl"),
            system_prompt=cfg.system_prompt or SYSTEM_PROMPT,
            split=cfg.test_split,
            max_new_tokens=cfg.test_max_new_tokens,
            batch_size=cfg.test_batch_size,
            max_examples=cfg.test_max_examples,
            hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
            repetition_penalty=cfg.test_repetition_penalty,
            no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
        )
        # source_dir == output_dir here (the just-saved final adapter/tokenizer
        # files, not a numbered checkpoint) -- update_best_checkpoint's ignore
        # patterns exist specifically so this doesn't copy output_dir into a
        # subdirectory of itself.
        update_best_checkpoint(cfg.output_dir, cfg.output_dir, final_metrics, trainer.state.global_step)

        if cfg.test_entity_dataset_id:
            run_test_eval(
                model,
                tokenizer,
                dataset_id=cfg.test_entity_dataset_id,
                input_column=cfg.test_input_column,
                target_column=cfg.test_target_column,
                output_path=str(output_dir / "test_eval_entity" / "predictions.jsonl"),
                system_prompt=cfg.system_prompt or SYSTEM_PROMPT,
                split=cfg.test_split,
                max_new_tokens=cfg.test_max_new_tokens,
                batch_size=cfg.test_batch_size,
                hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
                repetition_penalty=cfg.test_repetition_penalty,
                no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
            )
    if cfg.push_to_hub:
        sync_output_dir(cfg, commit_message="final")


if __name__ == "__main__":
    main()
