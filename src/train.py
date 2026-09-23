"""LoRA SFT for correcting Persian Whisper ASR output against frontier-model transcripts.

Data is expected to already be preprocessed by the data team and published as a
Hugging Face dataset with an input column (default: `text_whisper`, the in-house
Whisper output) and a target column (default: `text`, the frontier-model
transcript). This script only handles tokenization, LoRA setup, and training —
see data.py:load_sft_dataset for the (thin) loading step to swap in your
teammate's actual loading/post-processing call.

All hyperparameters live in a YAML config (see configs/train/base.yaml) rather than
CLI flags, so a run is fully reproducible from one file. Use --set for quick
one-off overrides without editing the file, e.g.:

    python train.py --config configs/train/base.yaml --set training.learning_rate=1e-4 --set lora.r=32

To queue up several runs back-to-back for comparison, use run_sweep.py instead
of calling this script directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from config import Config, load_config
from data import PadCollator, build_example, load_sft_dataset, resolve_system_prompt
from evaluate import TestEvalCallback, run_secondary_eval, run_test_eval, update_best_checkpoint
from hub_sync import SyncToHubCallback, repo_folder_name, sync_output_dir

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a YAML config file (see configs/train/base.yaml).")
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


# Known punctuated/punctuation-free target column names across the datasets
# this pipeline actually touches -- text_soniox (our own curated data),
# text_raw (ErfanRou/callcc-test-1k and the entity/typo eval slices), text
# (the punctuation-free default everywhere). Not exhaustive for an arbitrary
# custom dataset_id, just a sanity net for the common case.
_PUNCTUATED_COLUMNS = ("text_soniox", "text_raw")
_UNPUNCTUATED_COLUMNS = ("text",)


def warn_if_punctuation_flag_inconsistent(cfg: Config) -> None:
    """include_punctuation only swaps the system prompt -- target_column/
    test_target_column are set independently (the punctuated column has a
    different name per dataset), so it's easy to flip one and forget the
    other. That silently trains/evaluates against a system prompt that
    contradicts the actual target shape, so this warns (doesn't raise) at
    startup for the common, recognized column names.
    """
    for label, column in [("target_column", cfg.target_column), ("test_target_column", cfg.test_target_column)]:
        if cfg.include_punctuation and column in _UNPUNCTUATED_COLUMNS:
            print(
                f"WARNING: include_punctuation is true but {label}={column!r} looks "
                "punctuation-free -- did you mean text_soniox (local curated data) or "
                "text_raw (Hub datasets/eval slices)?",
                file=sys.stderr,
            )
        elif not cfg.include_punctuation and column in _PUNCTUATED_COLUMNS:
            print(
                f"WARNING: {label}={column!r} looks punctuated but include_punctuation "
                "is false -- the system prompt will still say not to add punctuation, "
                "contradicting the training/eval target. Set include_punctuation: true "
                "to match.",
                file=sys.stderr,
            )


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


class WeightedLossTrainer(Trainer):
    """Trainer subclass that multiplies each target token's loss by a
    per-token weight (batch key "token_weights", built by
    data.py's build_example()/PadCollator from Config.
    recoverable_correction_weight) instead of the model's default uniform
    cross-entropy.

    Only ever constructed when that weight isn't 1.0 (see main(), below) --
    a run that doesn't use this feature gets the plain `Trainer` untouched,
    so this class existing changes nothing about the already-working default
    path.

    Recomputes the causal-LM shift-and-cross-entropy by hand instead of
    letting the model compute its own loss from `labels`:
    AutoModelForCausalLM's built-in loss has no hook for per-token weights,
    so `labels` is withheld from the model call (labels=None -> it returns
    logits only, skipping its own loss computation) and the loss is computed
    here instead. With every weight equal to 1.0, this reduces to exactly
    the model's own default loss (a plain mean over non--100 positions) --
    verified directly against the built-in loss on a real batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This class computes its own loss without num_items_in_batch --
        # HF's own guidance (see Trainer.compute_loss's docstring) is to set
        # this False in that case, or gradient-accumulation loss scaling can
        # be slightly inaccurate.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        token_weights = inputs.pop("token_weights")
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard causal-LM shift: token i's logits predict token i+1.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = token_weights[..., 1:].contiguous()

        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        # Zero out weights at ignored (-100) positions explicitly -- their
        # loss is already 0 from ignore_index above, but a nonzero weight
        # there would still inflate the weighted-average denominator below.
        flat_weights = shift_weights.view(-1).to(per_token_loss.dtype) * (shift_labels.view(-1) != -100).to(
            per_token_loss.dtype
        )
        total_weight = flat_weights.sum().clamp(min=1e-8)
        loss = (per_token_loss * flat_weights).sum() / total_weight

        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=_parse_overrides(args.overrides))

    if cfg.push_to_hub and not cfg.hub_repo_id:
        raise ValueError("hub.push_to_hub is true but hub.repo_id is not set.")
    if cfg.on_long_example not in ("drop", "truncate"):
        raise ValueError(f"on_long_example must be 'drop' or 'truncate', got {cfg.on_long_example!r}")
    if cfg.lora_init_mode not in ("fresh", "continue", "merge_and_new"):
        raise ValueError(f"lora_init_mode must be 'fresh', 'continue', or 'merge_and_new', got {cfg.lora_init_mode!r}")
    if cfg.lora_init_mode == "fresh" and cfg.init_lora_from:
        raise ValueError(
            "init_lora_from is set but lora_init_mode is 'fresh' -- set lora_init_mode to "
            "'continue' or 'merge_and_new', or clear init_lora_from."
        )
    if cfg.lora_init_mode != "fresh" and not cfg.init_lora_from:
        raise ValueError(f"lora_init_mode={cfg.lora_init_mode!r} requires init_lora_from to be set.")
    if cfg.lora_init_mode == "merge_and_new" and cfg.use_unsloth:
        raise ValueError(
            "lora_init_mode='merge_and_new' is not supported together with use_unsloth "
            "(untested combination, not implemented) -- set use_unsloth: false."
        )
    if cfg.lora_init_mode != "fresh" and cfg.resume_from_checkpoint:
        # Two different starting strategies: resume_from_checkpoint restores
        # this run's own optimizer/scheduler/global_step (continuing an
        # interrupted run in place); lora_init_mode='continue'/'merge_and_new'
        # only seed a brand-new run's weights (fresh optimizer/step count,
        # possibly different data/hyperparameters/output entirely). Setting
        # both is almost certainly a mistake, not a real "use both" case.
        raise ValueError(
            "lora_init_mode != 'fresh' and resume_from_checkpoint are mutually exclusive -- "
            "resume_from_checkpoint continues THIS run's own trainer state; "
            "lora_init_mode='continue'/'merge_and_new' seed a NEW run's weights from a different adapter."
        )
    warn_if_punctuation_flag_inconsistent(cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Saved alongside checkpoints/tb logs so this run is self-describing --
    # the load_model notebook reads model_id/system_prompt back out of this.
    (output_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    # Deterministic path (not transformers' default output_dir/runs/<timestamp>)
    # so every run's TensorBoard logs land in the same place inside output_dir.
    tb_dir = cfg.tensorboard_logging_dir or str(output_dir / "tb")
    os.environ["TENSORBOARD_LOGGING_DIR"] = tb_dir

    if cfg.use_unsloth:
        # Unsloth's from_pretrained/get_peft_model replace both the plain
        # tokenizer/model loading and the LoraConfig/get_peft_model setup
        # below -- everything after this branch (Trainer, callbacks, eval)
        # is unchanged, since Unsloth's optimization is applied at the
        # model-loading step, not the training loop. `target_modules` is the
        # explicit projection-layer list from Unsloth's own docs (their
        # get_peft_model doesn't take peft's "all-linear" shorthand); this
        # is the standard LLaMA-family module set both Qwen and Gemma follow.
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_id,
            max_seq_length=cfg.max_length,
            load_in_4bit=cfg.load_in_4bit,
            full_finetuning=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
            max_seq_length=cfg.max_length,
        )
    else:
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

        if cfg.lora_init_mode == "merge_and_new":
            # Permanently folds an existing adapter into the base model
            # before any NEW LoRA is attached, so the new LoRA gets fresh
            # capacity instead of fighting for space in the same rank-r
            # matrices as everything already learned. PeftModel.from_pretrained
            # (not the manual LoraConfig+load_peft_weights path used for
            # "continue" below) reads the adapter's own saved adapter_config.json
            # to reconstruct its exact original shape -- required here since
            # this is a temporary wrapper purely for merging, not the run's
            # own LoRA config, so there's no reason it should have to match
            # cfg.lora_r/lora_alpha at all.
            from peft import PeftModel

            merge_kwargs = {"subfolder": cfg.init_lora_from_subfolder} if cfg.init_lora_from_subfolder else {}
            model = PeftModel.from_pretrained(model, cfg.init_lora_from, **merge_kwargs)
            model = model.merge_and_unload()
            print(f"Merged existing LoRA from {cfg.init_lora_from} (subfolder={cfg.init_lora_from_subfolder}) into the base model.")

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

    if cfg.lora_init_mode == "continue":
        # Seeds this brand-new run's LoRA weights from an existing adapter
        # (a local path, or a Hub repo id -- optionally with
        # init_lora_from_subfolder for a repo that holds several runs'
        # adapters in per-run subfolders, matching this project's own Hub
        # layout) -- NOT the same thing as resume_from_checkpoint (see the
        # validation above): only the adapter weights are loaded, not
        # optimizer/scheduler/global_step, so this run starts at step 0 with
        # its own (possibly entirely different) data/hyperparameters/output,
        # just not from LoRA's usual zero-initialized B matrix. Uses this
        # run's OWN lora_r/lora_alpha (via get_peft_model above), not
        # whatever shape the source adapter happens to have -- a mismatch is
        # a loud, explicit error below, not a silent fallback to the
        # adapter's own shape (unlike merge_and_new's PeftModel.from_pretrained,
        # where there's no "this run's own shape" yet to conflict with).
        from peft import load_peft_weights, set_peft_model_state_dict

        load_kwargs = {}
        if cfg.init_lora_from_subfolder:
            load_kwargs["subfolder"] = cfg.init_lora_from_subfolder
        state_dict = load_peft_weights(cfg.init_lora_from, **load_kwargs)
        mismatch_msg = (
            f"init_lora_from={cfg.init_lora_from!r} (subfolder={cfg.init_lora_from_subfolder!r}) "
            f"doesn't match this run's LoRA shape -- lora_r/lora_alpha/target_modules must match "
            f"the adapter being loaded."
        )
        try:
            load_result = set_peft_model_state_dict(model, state_dict)
        except RuntimeError as e:
            # A same-name-different-shape mismatch (e.g. a different lora_r)
            # raises directly here rather than returning gracefully --
            # verified directly against a real load, not assumed.
            raise ValueError(f"{mismatch_msg} Underlying error: {e}") from e
        # missing_keys is near-useless as a mismatch signal here: it's
        # dominated by every frozen base-model weight (base_layer.weight,
        # embed_tokens, lm_head, ...), which are never part of a saved LoRA
        # adapter's state dict at all and are "missing" even on a perfectly
        # successful load -- verified directly (21 such keys on a trivial
        # matching-shape load). Only a *lora_*-named missing key, or any
        # unexpected_keys (a saved key that doesn't match this model's
        # target_modules at all, e.g. a different module set), is a real
        # problem worth failing on.
        lora_missing = [k for k in load_result.missing_keys if "lora_" in k]
        if lora_missing or load_result.unexpected_keys:
            raise ValueError(
                f"{mismatch_msg} missing LoRA keys={lora_missing}, unexpected_keys={load_result.unexpected_keys}"
            )
        print(f"Initialized LoRA weights from {cfg.init_lora_from} (subfolder={cfg.init_lora_from_subfolder})")

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
    masked_low_signal_tokens_total = 0
    masked_low_signal_examples = 0
    weighted_recoverable_tokens_total = 0
    weighted_recoverable_examples = 0
    system_prompt = resolve_system_prompt(cfg)

    # Word frequencies over the training targets. Used by three separate
    # features, so it's built unconditionally (one pass over text already in
    # memory): mask_low_signal_corrections and recoverable_correction_weight
    # both feed it to word_correction_categories() to decide, per word,
    # whether a Whisper error is a genuine recoverable correction (weighted
    # up) or unrecoverable (masked from the *training loss*, only when
    # enabled) -- and evaluate.py's fix_rate_recoverable drops the same
    # unrecoverable corrections from its *eval* denominator (always -- it's
    # the only frequency source that makes that metric comparable across
    # runs, since the eval set alone is too small to tell a rare name from a
    # merely eval-rare common word).
    corpus_freq = Counter()
    for text in raw["train"][cfg.target_column]:
        corpus_freq.update(text.split())
    # Only passed to build_example when at least one of the two features
    # that use it is active -- both rely on word_correction_categories()'s
    # corpus-frequency gate to draw the same category-2/3 line
    # low_signal_word_ranges() itself uses, so a run using ONLY
    # recoverable_correction_weight (masking off) still needs this, not just
    # a run using mask_low_signal_corrections.
    uses_word_categories = cfg.mask_low_signal_corrections or cfg.recoverable_correction_weight != 1.0
    word_category_corpus_freq = corpus_freq if uses_word_categories else None

    def _map(example):
        result = build_example(
            tokenizer,
            example[cfg.input_column],
            example[cfg.target_column],
            max_length=cfg.max_length,
            on_long_example=cfg.on_long_example,
            system_prompt=system_prompt,
            mask_low_signal_corrections=cfg.mask_low_signal_corrections,
            mask_min_similarity=cfg.mask_min_similarity,
            mask_corpus_freq=word_category_corpus_freq,
            mask_max_common_freq=cfg.mask_max_common_freq,
            recoverable_correction_weight=cfg.recoverable_correction_weight,
        )
        length_stats[result["status"]] += 1
        nonlocal masked_low_signal_tokens_total, masked_low_signal_examples
        nonlocal weighted_recoverable_tokens_total, weighted_recoverable_examples
        if result["masked_low_signal_tokens"]:
            masked_low_signal_tokens_total += result["masked_low_signal_tokens"]
            masked_low_signal_examples += 1
        if result["weighted_recoverable_tokens"]:
            weighted_recoverable_tokens_total += result["weighted_recoverable_tokens"]
            weighted_recoverable_examples += 1
        return result

    tokenized = raw.map(_map, remove_columns=raw["train"].column_names)

    n_total = sum(length_stats.values())
    print(
        f"Tokenization ({n_total} rows, train+eval combined): "
        f"{length_stats['ok']} ok, {length_stats['truncated']} truncated, "
        f"{length_stats['dropped']} dropped (exceeded max_length={cfg.max_length})"
    )
    if cfg.mask_low_signal_corrections:
        print(
            f"Low-signal correction masking (min_similarity={cfg.mask_min_similarity}, "
            f"max_common_freq={cfg.mask_max_common_freq}): "
            f"{masked_low_signal_tokens_total} target tokens masked out of loss across "
            f"{masked_low_signal_examples} examples"
        )
    if cfg.recoverable_correction_weight != 1.0:
        print(
            f"Recoverable-correction up-weighting (weight={cfg.recoverable_correction_weight}): "
            f"{weighted_recoverable_tokens_total} target tokens up-weighted across "
            f"{weighted_recoverable_examples} examples"
        )
    if length_stats["dropped"]:
        tokenized = tokenized.filter(lambda ex: ex["status"] != "dropped")
    tokenized = tokenized.remove_columns(["status", "masked_low_signal_tokens", "weighted_recoverable_tokens"])
    if cfg.recoverable_correction_weight == 1.0:
        tokenized = tokenized.remove_columns(["token_weights"])

    collator = PadCollator(
        pad_token_id=tokenizer.pad_token_id,
        include_token_weights=cfg.recoverable_correction_weight != 1.0,
    )

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
        # When use_unsloth is on, get_peft_model() above was already called
        # with use_gradient_checkpointing="unsloth" -- Trainer's own generic
        # gradient_checkpointing_enable() is a different implementation, so
        # leaving this True too would double up (or conflict with) Unsloth's
        # own checkpointing rather than complementing it.
        gradient_checkpointing=not cfg.use_unsloth,
        # Batches sequences of similar length together (still shuffled at the
        # mega-batch level, not a fixed dataset order) instead of forming
        # batches in dataset order -- this dataset's length spread is large
        # (chunked rows ~15-40 words vs. assembled rows up to ~3,000), so an
        # unsorted batch can pad every short sequence up to whatever long one
        # happens to land next to it, wasting real compute/memory on pad
        # tokens. `group_by_length` (the old boolean flag) was renamed to
        # this string field in the installed transformers version -- verified
        # directly, the old kwarg name raises TypeError here.
        train_sampling_strategy="group_by_length",
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
        test_eval_callback = TestEvalCallback(cfg, model, tokenizer, corpus_freq)
        callbacks.append(test_eval_callback)
    if cfg.push_to_hub:
        # Registered after the test-eval callback: callbacks fire in list
        # order, so each save's freshly written test_eval/ files are already
        # on disk by the time this uploads output_dir.
        callbacks.append(SyncToHubCallback(cfg))

    # WeightedLossTrainer only when actually needed -- see its docstring --
    # so the default (weight 1.0 everywhere) path is the plain `Trainer`,
    # unchanged by this feature existing.
    trainer_cls = WeightedLossTrainer if cfg.recoverable_correction_weight != 1.0 else Trainer
    trainer = trainer_cls(
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
            system_prompt=resolve_system_prompt(cfg),
            split=cfg.test_split,
            max_new_tokens=cfg.test_max_new_tokens,
            batch_size=cfg.test_batch_size,
            max_examples=cfg.test_max_examples,
            hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
            repetition_penalty=cfg.test_repetition_penalty,
            no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
            fix_weight=cfg.test_fix_weight,
            low_signal_corpus_freq=corpus_freq,
            low_signal_min_similarity=cfg.mask_min_similarity,
            low_signal_max_common_freq=cfg.mask_max_common_freq,
        )
        print(f"Baseline: wer={baseline_metrics['wer']}, exact_match={baseline_metrics['exact_match']}, "
              f"hallucination_rate={baseline_metrics['hallucination_rate']}, "
              f"fix_rate={baseline_metrics['fix_rate']}, preservation_rate={baseline_metrics['preservation_rate']}")
        if baseline_metrics["wer"] is not None:
            trainer.log({
                "test_wer": baseline_metrics["wer"],
                "test_wer_zwnj_normalized": baseline_metrics["wer_zwnj_normalized"],
                "test_exact_match": baseline_metrics["exact_match"],
                "test_hallucination_rate": baseline_metrics["hallucination_rate"],
                "test_fix_rate": baseline_metrics["fix_rate"],
                "test_preservation_rate": baseline_metrics["preservation_rate"],
                "test_targeted_score": baseline_metrics["targeted_score"],
                "test_fix_rate_lenient": baseline_metrics["fix_rate_lenient"],
                "test_preservation_rate_lenient": baseline_metrics["preservation_rate_lenient"],
                "test_targeted_score_lenient": baseline_metrics["targeted_score_lenient"],
            })

        for name, ds_id, subdir in [
            ("entity", cfg.test_entity_dataset_id, "test_eval_entity_baseline"),
            ("typo", cfg.test_typo_dataset_id, "test_eval_typo_baseline"),
        ]:
            secondary_metrics = run_secondary_eval(
                model, tokenizer, cfg, ds_id, name, subdir, trainer, corpus_freq)
            if secondary_metrics is not None:
                print(f"Baseline ({name} slice): wer={secondary_metrics['wer']}, "
                      f"exact_match={secondary_metrics['exact_match']}")

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
            system_prompt=resolve_system_prompt(cfg),
            split=cfg.test_split,
            max_new_tokens=cfg.test_max_new_tokens,
            batch_size=cfg.test_batch_size,
            max_examples=cfg.test_max_examples,
            hallucination_overlap_floor=cfg.test_hallucination_overlap_floor,
            repetition_penalty=cfg.test_repetition_penalty,
            no_repeat_ngram_size=cfg.test_no_repeat_ngram_size,
            fix_weight=cfg.test_fix_weight,
            low_signal_corpus_freq=corpus_freq,
            low_signal_min_similarity=cfg.mask_min_similarity,
            low_signal_max_common_freq=cfg.mask_max_common_freq,
        )
        # source_dir == output_dir here (the just-saved final adapter/tokenizer
        # files, not a numbered checkpoint) -- update_best_checkpoint's ignore
        # patterns exist specifically so this doesn't copy output_dir into a
        # subdirectory of itself.
        update_best_checkpoint(cfg.output_dir, cfg.output_dir, final_metrics, trainer.state.global_step)

        run_secondary_eval(model, tokenizer, cfg, cfg.test_entity_dataset_id, "entity",
                            "test_eval_entity", None, corpus_freq)
        run_secondary_eval(model, tokenizer, cfg, cfg.test_typo_dataset_id, "typo",
                            "test_eval_typo", None, corpus_freq)
    if cfg.push_to_hub:
        sync_output_dir(cfg, commit_message="final")


if __name__ == "__main__":
    main()
