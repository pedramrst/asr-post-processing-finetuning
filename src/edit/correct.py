#!/usr/bin/env python3
"""The end-to-end corrector: raw Whisper text in, corrected text out.

Runs the whole edit-based pipeline in one process, with no external API
calls -- the LLM judges (MiMo/DeepSeek) were only ever used offline to build
the verifier's training data and to audit it, and play no part here:

  1. rules.py       deterministic fixes (glued words, stutters). No model.
  2. rewrite model  a LoRA adapter over Qwen3.5-2B, used ONLY as an error
                    detector: it generates a full rewrite, which is never
                    shown to anyone -- we just diff it against the Whisper
                    text to get candidate edits. A rewrite that stops early
                    (premature_stop, evaluate.py's rule) contributes none.
  3. filtering      candidates must be substitutions of <= max_span_words
                    words, and never style-only, filler-only, or
                    number-changing ones (edits.py's changes_number).
  4. verifier       a second LoRA adapter over the SAME base model, scoring
                    each candidate on its own: P(YES)/(P(YES)+P(NO)) of the
                    first answer token, one forward pass, no generation.
  5. apply          candidates scoring >= --threshold are applied to the
                    Whisper text by edits.py's validated apply_edits.

Every word outside an applied edit is Whisper's own. Both adapters sit on
one shared base model (same Qwen3.5-2B, same LoRA shape), so only one copy
of the base weights is loaded.

Input: a .jsonl with a text field (--input-column), or --dataset for a Hub
dataset. `crm_names` per row is optional and only used as context.

  python3 src/edit/correct.py --input transcripts.jsonl --output corrected.jsonl
  python3 src/edit/correct.py --dataset ErfanRou/callcc-test-1k --limit 50 --threshold 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

import torch  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from corpus import load_corpus_freq  # noqa: E402
from data import SYSTEM_PROMPT as REWRITE_SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_PUNCTUATION  # noqa: E402
from edits import apply_edits, validate_edits, Edit  # noqa: E402
from evaluate import _load_hub_columns_pruned, generate_batch  # noqa: E402
from grounding import crm_candidates  # noqa: E402
from rules import rule_edits  # noqa: E402
from train_edit import chat_prompts  # noqa: E402
from train_verifier import SYSTEM_PROMPT as VERIFIER_SYSTEM_PROMPT, extract_candidates, render_input  # noqa: E402

load_dotenv()

REWRITE_RUN = "qwen3.5-2b-100pct-masked-weighted-v2"
VERIFIER_RUN = "verifier-qwen3.5-2b-v2"
HUB_REPO = "PedramR/ASR_Post-processing"
# evaluate.py's premature-stop rule: a rewrite far shorter than its input has
# stopped early and its "edits" are a truncation, not corrections.
PREMATURE_STOP_RATIO = 0.3
PREMATURE_STOP_MIN_INPUT_WORDS = 30


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Local .jsonl, one row per transcript.")
    src.add_argument("--dataset", help="Hub dataset id instead (uses --split).")
    p.add_argument("--split", default="test")
    p.add_argument("--input-column", default="text_whisper")
    p.add_argument("--output", default="corrected.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Verifier score to accept a candidate. 0.8 ~= 92%% judged precision, 0.7 ~= 81%% "
                        "(see README's step 4); higher is more conservative.")
    p.add_argument("--hub-repo", default=HUB_REPO)
    p.add_argument("--rewrite-run", default=REWRITE_RUN)
    p.add_argument("--verifier-run", default=VERIFIER_RUN)
    p.add_argument("--base-model", default=None, help="Default: read from the rewrite run's adapter_config.json.")
    p.add_argument("--corpus-freq-repo", default="PedramR/ASR_Post-processing-dataset")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-span-words", type=int, default=4)
    p.add_argument("--context-words", type=int, default=12)
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Also write after_rules, the raw rewrite, and every candidate with its score.")
    return p.parse_args()


def load_models(args):
    """Base model once, with both adapters attached under their own names."""
    base_id = args.base_model
    if not base_id:
        base_id = json.load(open(hf_hub_download(args.hub_repo, f"{args.rewrite_run}/adapter_config.json")
                                 ))["base_model_name_or_path"]
    print(f"Base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    # Both adapters share these base weights -- one ~4GB copy, not two.
    model = PeftModel.from_pretrained(model, args.hub_repo, subfolder=args.rewrite_run, adapter_name="rewrite")
    model.load_adapter(args.hub_repo, subfolder=args.verifier_run, adapter_name="verifier")
    model.eval()
    print(f"Adapters: rewrite={args.rewrite_run}, verifier={args.verifier_run}")
    return model, tokenizer


def rewrite_system_prompt(args) -> str:
    """The prompt the rewrite run was actually trained/evaluated with."""
    cfg = yaml.safe_load(open(hf_hub_download(args.hub_repo, f"{args.rewrite_run}/config.yaml")))
    if cfg.get("system_prompt"):
        return cfg["system_prompt"]
    return SYSTEM_PROMPT_WITH_PUNCTUATION if cfg.get("include_punctuation") else REWRITE_SYSTEM_PROMPT


def premature_stop(source: str, rewrite: str) -> bool:
    n_in = len(source.split())
    return n_in >= PREMATURE_STOP_MIN_INPUT_WORDS and len(rewrite.split()) < PREMATURE_STOP_RATIO * n_in


@torch.no_grad()
def score_candidates(model, tokenizer, items: list[dict], batch_size: int) -> list[float]:
    """P(YES)/(P(YES)+P(NO)) per candidate, from one forward pass each."""
    if not items:
        return []
    from decode_sweep import first_token_probs

    yes_id = tokenizer("YES", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer("NO", add_special_tokens=False)["input_ids"][0]
    prompts = chat_prompts(tokenizer, [render_input(it) for it in items], VERIFIER_SYSTEM_PROMPT)
    probs = first_token_probs(model, tokenizer, prompts, [yes_id, no_id], batch_size)
    return [y / (y + n) if y + n > 0 else 0.0 for y, n in probs]


def load_rows(args) -> list[dict]:
    if args.input:
        rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    else:
        columns = [args.input_column] + (["crm_metadata"] if args.dataset else [])
        ds = _load_hub_columns_pruned(args.dataset, columns, args.split)
        rows = [dict(r) for r in ds]
    rows = [r for r in rows if (r.get(args.input_column) or "").strip()]
    return rows[:args.limit] if args.limit else rows


def main():
    args = parse_args()
    rows = load_rows(args)
    print(f"{len(rows)} transcripts")
    names = [sorted(crm_candidates(r["crm_metadata"])) if r.get("crm_metadata") else (r.get("crm_names") or [])
             for r in rows]
    whisper = [r[args.input_column] for r in rows]

    # 1. rules.py -- deterministic, no model (allow_numbers: it only ever
    #    re-segments the same characters; see build_diffs.py's comment).
    freq = load_corpus_freq(args.corpus_freq_repo)
    after_rules = [apply_edits(w, rule_edits(w, freq, protected=set(n)), allow_numbers=True)
                   for w, n in zip(whisper, names)]

    model, tokenizer = load_models(args)
    started = time.time()

    # 2. rewrite model -> full rewrites, used only as an error detector.
    model.set_adapter("rewrite")
    rewrites = generate_batch(model, tokenizer,
                              chat_prompts(tokenizer, after_rules, rewrite_system_prompt(args)),
                              args.max_new_tokens, args.batch_size)

    # 3. candidates: substitutions only, no style/filler/number (extract_candidates),
    #    and none at all from a rewrite that stopped early.
    stopped = [premature_stop(a, rw) for a, rw in zip(after_rules, rewrites)]
    cands = [[] if st else extract_candidates(a, rw, n, args.context_words, args.max_span_words)
             for a, rw, n, st in zip(after_rules, rewrites, names, stopped)]
    flat = [(i, c) for i, cs in enumerate(cands) for c in cs]
    print(f"{len(flat)} candidates from {len(rows) - sum(stopped)} rewrites "
          f"({sum(stopped)} skipped: stopped early)")

    # 4. verifier -- one forward pass per candidate, no generation.
    model.set_adapter("verifier")
    for (i, c), s in zip(flat, score_candidates(model, tokenizer, [c for _, c in flat], args.batch_size)):
        c["score"] = s

    # 5. apply the accepted ones to WHISPER's words (after rules), nothing else.
    n_applied = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            accepted = [Edit(c["start"], c["end"], c["original"].split(), c["proposed"].split())
                        for c in cands[i] if c["score"] >= args.threshold]
            valid, rejected = validate_edits(after_rules[i].split(), sorted(accepted, key=lambda e: e.start))
            corrected = apply_edits(after_rules[i], valid)
            n_applied += len(valid)
            out = {"input": whisper[i], "output": corrected,
                   "edits": [{"start": e.start, "end": e.end, "original": " ".join(e.original),
                              "replacement": " ".join(e.replacement)} for e in valid]}
            if args.keep_intermediates:
                out.update(after_rules=after_rules[i], rewrite=rewrites[i], premature_stop=stopped[i],
                           candidates=cands[i], rejected=[reason for _, reason in rejected])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    elapsed = time.time() - started
    changed = sum(1 for i in range(len(rows)) if any(c["score"] >= args.threshold for c in cands[i]))
    print(f"Applied {n_applied} edits to {changed}/{len(rows)} transcripts at threshold {args.threshold} "
          f"({len(flat) - n_applied} candidates rejected)")
    print(f"{elapsed:.1f}s for {len(rows)} transcripts ({elapsed / max(len(rows), 1):.2f}s each) -> {args.output}")


if __name__ == "__main__":
    main()
