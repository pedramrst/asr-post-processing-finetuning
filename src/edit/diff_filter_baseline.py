#!/usr/bin/env python3
"""Step 1 of the edit-based method: the no-training baseline.

Takes an existing full-rewrite run's test predictions (by default the
*original*, not-fine-tuned Qwen3.5-2b -- the `test_eval*_baseline` files a
run logs before training starts), turns each rewrite back into word-level
edits against the Whisper input (edits.extract_edits), keeps only the
edits that pass a filter, applies them to the Whisper text (so every
unaccepted word is Whisper's own), and scores the result.

Compared per eval slice (main test set, entity slice, typo slice):

  whisper          untouched Whisper -- the floor every method must beat
  rewrite_raw      the model's full rewrite exactly as generated
  rewrite          all edits = the rewrite with punctuation stripped (see
                   strip_punctuation; every filter below starts from these)
  oracle           only edits that individually help against the reference
                   -- the ceiling of *any* filter over this model's edits
  no_style         drops edits that are only spelling/dialect variants
                   (persian_normalize lenient-equivalent)
  phonetic@T       non-style substitutions of <= 3 words whose two sides are
                   phonetically similar (>= T) -- "misrecognition-shaped"
  crm_in_rewrite   only edits that introduce a CRM name word
  crm_snap         model-free: snap rare near-miss words onto CRM names
  rules            model-free: split glued words, undo stutter-doubled
                   letters (rules.py)
  rules+crm_snap, phonetic@0.8+rules+crm_snap   combinations

Model-free edits are logged with their individual effect in
<slice>/model_free_edits.jsonl.

Before reporting anything, re-scores the unfiltered predictions and checks
they reproduce the run's own metrics.json, so these numbers are comparable
with the rewrite method's.

Writes <output-dir>/<slice>/summary.json, a combined summary.md table, and
<slice>/edits.jsonl (every extracted edit with its features and whether it
helped or hurt on its own) for inspecting what the model actually edits.

Example:
  python3 src/edit/diff_filter_baseline.py
  python3 src/edit/diff_filter_baseline.py --run qwen3.5-2b --slices main
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402

from corpus import load_corpus_freq  # noqa: E402
from edits import Edit, apply_edits, extract_edits, validate_edits  # noqa: E402
from evaluate import _load_hub_columns_pruned  # noqa: E402
from grounding import (  # noqa: E402
    crm_candidates, crm_snap_edits, load_gemini_entities, reference_entity_spans, span_similarity,
)
from normalize import is_style_edit, strip_punctuation  # noqa: E402
from rules import rule_edits  # noqa: E402
from scoring import lenient_positions, score  # noqa: E402

load_dotenv()

SLICES = {  # slice name -> subfolder of the run in the experiments repo
    "main": "test_eval_baseline",
    "entity": "test_eval_entity_baseline",
    "typo": "test_eval_typo_baseline",
}
REPRODUCED_KEYS = [
    "wer", "wer_zwnj_normalized", "exact_match", "fix_rate", "preservation_rate",
    "fix_rate_lenient", "preservation_rate_lenient", "fix_rate_recoverable", "hallucination_rate",
]
TABLE_KEYS = [
    ("wer", "WER"), ("fix_rate_lenient", "fix (len.)"), ("preservation_rate_lenient", "keep (len.)"),
    ("fixed_lenient", "fixed"), ("broken_lenient", "broken"), ("net_fixed_lenient", "net"),
    ("broken_per_100_correct", "broken/100"), ("entity_fix_rate", "entity fix"),
    ("entity_keep_rate", "entity keep"), ("rows_changed", "rows chg"),
]
MAX_SPAN_WORDS = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hub-repo", default="PedramR/ASR_Post-processing")
    p.add_argument("--run", default="qwen3.5-2b-50pct-masked-weighted",
                   help="Run folder whose *_baseline predictions (the model before fine-tuning) to filter. "
                        "Pass --subdir-suffix '' to use its fine-tuned predictions instead.")
    p.add_argument("--subdir-suffix", default="_baseline")
    p.add_argument("--slices", default="main,entity,typo")
    p.add_argument("--test-repo", default="ErfanRou/callcc-test-1k",
                   help="Source of crm_metadata/call_id, joined to predictions on the Whisper text.")
    p.add_argument("--train-repo", default="PedramR/ASR_Post-processing-dataset",
                   help="Training data whose `text` column gives the corpus word frequencies (same source "
                        "train.py uses for fix_rate_recoverable).")
    p.add_argument("--entities-dir", default="data/output-backup")
    p.add_argument("--output-dir", default="outputs/edit/diff_filter_baseline")
    return p.parse_args()


def annotate(e: Edit, candidates: set[str]) -> Edit:
    e.meta.update(
        kind=e.kind,
        style=is_style_edit(e),
        similarity=round(span_similarity(e.original, e.replacement), 3) if e.original and e.replacement else 0.0,
        span_words=max(len(e.original), len(e.replacement)),
        crm_hit=any(w in candidates and w not in e.original for w in e.replacement),
    )
    return e


def edit_effect(src: str, ref: str, e: Edit, base: tuple[int, int]) -> tuple[int, int]:
    """(lenient words fixed, lenient words broken) by applying `e` alone."""
    target, already, still, _ = lenient_positions(src, apply_edits(src, [e]), ref)
    return len(target & still) - base[0], base[1] - len(already & still)


def phonetic(threshold):
    return lambda e: (not e.meta["style"] and e.kind == "substitute"
                      and e.meta["span_words"] <= MAX_SPAN_WORDS and e.meta["similarity"] >= threshold)


FILTERS = {
    "whisper": lambda e: False,
    "rewrite": lambda e: True,
    "oracle": lambda e: e.meta["net_effect"] > 0,
    "no_style": lambda e: not e.meta["style"],
    "phonetic@0.5": phonetic(0.5),
    "phonetic@0.67": phonetic(0.67),
    "phonetic@0.8": phonetic(0.8),
    "crm_in_rewrite": lambda e: e.meta["crm_hit"],
}
# Methods that add model-free edits (grounding.crm_snap_edits, rules.rule_edits),
# optionally on top of a filtered subset of the model's own edits. Earlier
# proposers win overlaps.
COMBINED = {
    "crm_snap": (["crm_snap"], None),
    "rules": (["rules"], None),
    "rules+crm_snap": (["crm_snap", "rules"], None),
    "phonetic@0.8+rules+crm_snap": (["crm_snap", "rules"], "phonetic@0.8"),
}


def run_slice(name, rows, corpus_freq, test_by_whisper, gemini, out_dir: Path) -> dict:
    sources = [r["input"] for r in rows]
    references = [r["reference"] for r in rows]
    outputs = [strip_punctuation(r["output"]) for r in rows]

    joined = [test_by_whisper.get(s) for s in sources]
    missing = sum(j is None for j in joined)
    if missing:
        print(f"  [{name}] {missing}/{len(rows)} rows have no test-set match -- no CRM/entity info for them")
    candidates = [crm_candidates(j["crm_metadata"]) if j else set() for j in joined]
    entity_spans = [
        reference_entity_spans(ref, crm_candidates(j["crm_metadata"]) if j else set(),
                               gemini.get(j["call_id"], []) if j else [])
        for ref, j in zip(references, joined)
    ]

    # Rewrite -> edits -> rewrite must round-trip, or everything below is off.
    row_edits = []
    with open(out_dir / "edits.jsonl", "w", encoding="utf-8") as f:
        for src, out, ref, cands in tqdm(list(zip(sources, outputs, references, candidates)),
                                         desc=f"{name}: edits", leave=False):
            edits = [annotate(e, cands) for e in extract_edits(src, out)]
            assert apply_edits(src, edits).split() == out.split(), "edit round-trip failed"
            target, already, still, _ = lenient_positions(src, src, ref)
            base = (len(target & still), len(already & still))
            for e in edits:
                fixed, broken = edit_effect(src, ref, e, base)
                e.meta.update(fixed=fixed, broken=broken, net_effect=fixed - broken)
                f.write(json.dumps({"original": " ".join(e.original), "replacement": " ".join(e.replacement),
                                    "start": e.start, "end": e.end, **e.meta}, ensure_ascii=False) + "\n")
            row_edits.append(edits)

    # The model's rewrite exactly as it was generated, punctuation included.
    results = {"rewrite_raw": score(sources, [r["output"] for r in rows], references, corpus_freq, entity_spans)}
    for fname, keep in FILTERS.items():
        hyps = [apply_edits(s, [e for e in es if keep(e)]) for s, es in zip(sources, row_edits)]
        results[fname] = score(sources, hyps, references, corpus_freq, entity_spans)

    proposers = {
        "crm_snap": [crm_snap_edits(s, c, corpus_freq) for s, c in zip(sources, candidates)],
        "rules": [rule_edits(s, corpus_freq, protected=c) for s, c in zip(sources, candidates)],
    }
    with open(out_dir / "model_free_edits.jsonl", "w", encoding="utf-8") as f:
        for pname, per_row in proposers.items():
            for src, ref, edits in zip(sources, references, per_row):
                if not edits:
                    continue
                target, already, still, _ = lenient_positions(src, src, ref)
                base = (len(target & still), len(already & still))
                for e in edits:
                    fixed, broken = edit_effect(src, ref, e, base)
                    f.write(json.dumps({"proposer": pname, "original": " ".join(e.original),
                                        "replacement": " ".join(e.replacement), "fixed": fixed,
                                        "broken": broken, **e.meta}, ensure_ascii=False) + "\n")

    for fname, (names, base_filter) in COMBINED.items():
        hyps = []
        for i, (s, es) in enumerate(zip(sources, row_edits)):
            proposed = [e for p in names for e in proposers[p][i]]
            if base_filter:
                proposed += [e for e in es if FILTERS[base_filter](e)]
            valid, _ = validate_edits(s.split(), proposed)
            hyps.append(apply_edits(s, valid))
        results[fname] = score(sources, hyps, references, corpus_freq, entity_spans)

    all_edits = [e for es in row_edits for e in es]
    results["_edit_stats"] = {
        "edits": len(all_edits),
        "edits_per_row": len(all_edits) / len(rows),
        "helpful": sum(e.meta["net_effect"] > 0 for e in all_edits),
        "harmful": sum(e.meta["net_effect"] < 0 for e in all_edits),
        "neutral": sum(e.meta["net_effect"] == 0 for e in all_edits),
        "style": sum(e.meta["style"] for e in all_edits),
        "by_kind": dict(Counter(e.kind for e in all_edits)),
        "rows_with_crm_candidates": sum(bool(c) for c in candidates),
        "rows_with_entity_spans": sum(bool(s) for s in entity_spans),
    }
    return results


def check_reproduction(name, ours: dict, logged: dict):
    diffs = {k: (ours.get(k), logged.get(k)) for k in REPRODUCED_KEYS
             if logged.get(k) is not None and abs((ours.get(k) or 0) - logged[k]) > 1e-9}
    if diffs:
        raise SystemExit(f"[{name}] re-scoring doesn't reproduce the run's metrics.json: {diffs}")
    print(f"  [{name}] re-scoring reproduces metrics.json ({len(REPRODUCED_KEYS)} metrics)")


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, int):
        return str(v)
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    run_cfg = yaml.safe_load(Path(hf_hub_download(args.hub_repo, f"{args.run}/resolved_config.yaml")).read_text())
    corpus_freq = load_corpus_freq(args.train_repo, run_cfg.get("train_fraction"),
                                   (run_cfg.get("training") or {}).get("seed", 42))
    test = _load_hub_columns_pruned(args.test_repo, ["call_id", "text_whisper", "crm_metadata"], "test")
    test_by_whisper = {r["text_whisper"]: r for r in test if r["text_whisper"]}
    gemini = load_gemini_entities(args.entities_dir) if Path(args.entities_dir).exists() else {}
    if not gemini:
        print(f"No Gemini entity files at {args.entities_dir} -- entity metrics use CRM names only")

    md = [f"# Diff-and-filter baseline: `{args.run}` ({args.subdir_suffix or 'fine-tuned'} predictions)\n"]
    for name in args.slices.split(","):
        subdir = SLICES[name].replace("_baseline", args.subdir_suffix)
        prefix = f"{args.run}/{subdir}"
        rows = [json.loads(l) for l in open(hf_hub_download(args.hub_repo, f"{prefix}/predictions.jsonl"),
                                           encoding="utf-8")]
        logged = json.loads(Path(hf_hub_download(args.hub_repo, f"{prefix}/metrics.json")).read_text())
        check_reproduction(name, score([r["input"] for r in rows], [r["output"] for r in rows],
                                       [r["reference"] for r in rows], corpus_freq), logged)

        slice_dir = out_root / name
        slice_dir.mkdir(parents=True, exist_ok=True)
        results = run_slice(name, rows, corpus_freq, test_by_whisper, gemini, slice_dir)
        (slice_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

        stats = results.pop("_edit_stats")
        md.append(f"\n## {name} ({len(rows)} rows)\n")
        md.append(f"{stats['edits']} edits ({stats['edits_per_row']:.1f}/row): {stats['helpful']} helpful, "
                  f"{stats['harmful']} harmful, {stats['neutral']} neutral; {stats['style']} style-only. "
                  f"Kinds: {stats['by_kind']}. Rows with CRM names: {stats['rows_with_crm_candidates']}; "
                  f"with reference entities: {stats['rows_with_entity_spans']}.\n")
        md.append("| method | " + " | ".join(h for _, h in TABLE_KEYS) + " |")
        md.append("|---" * (len(TABLE_KEYS) + 1) + "|")
        for method, m in results.items():
            md.append(f"| {method} | " + " | ".join(fmt(m.get(k)) for k, _ in TABLE_KEYS) + " |")
    (out_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
