#!/usr/bin/env python3
"""Step 2d: measure how far each LLM judge can be trusted, against the hand
labels from make_label_sheet.py's pages.

The number that matters most is **real_error precision**: of the
differences a judge calls a genuine Whisper error (so they'd become
correction targets), how many the person agreed were. A low-precision
judge puts wrong corrections into the training data -- the over-editing
this method is meant to avoid. Recall matters less: a missed real error
just stays uncorrected (Whisper's own text is kept).

Also reports overall agreement, a confusion matrix, per-stratum agreement,
the same numbers for the "all judges agree" rule (an edit becomes a target
only if every judge calls it real_error), each prompt version side by side,
and -- when the round-1 item file exists -- what build_diffs.py's filters
removed from the round-1 items and how the person had labelled those.

Labels are always recomputed from the raw verdicts (llm_judge.to_label), so
round 1's "same" and v2's "either" count as the same answer.

Example:
  python3 src/edit/calibrate.py
  python3 src/edit/calibrate.py --labels outputs/edit/targets/calibration_labels.json,outputs/edit/targets/calibration_labels_review.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from llm_judge import DEFAULT_MODELS, PROMPT_VERSION, cache_path, load_cache, to_label

LABELS = ["real_error", "soniox_wrong", "either_fine", "both_wrong", "uncertain"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", default="outputs/edit/targets/calibration_items.jsonl")
    p.add_argument("--round1-items", default="outputs/edit/targets/calibration_items_round1.jsonl")
    p.add_argument("--diffs", default="outputs/edit/targets/diffs.jsonl")
    p.add_argument("--labels", default="outputs/edit/targets/calibration_labels.json,"
                   "outputs/edit/targets/calibration_labels_review.json",
                   help="Comma-separated label files; later files override earlier ones (e.g. a review round).")
    p.add_argument("--judge-dir", default="outputs/edit/judge")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--prompt-versions", default=f"v1,{PROMPT_VERSION}")
    p.add_argument("--output", default="outputs/edit/targets/calibration_report.md")
    return p.parse_args()


def merge_labels(paths: list[str]) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for path in paths:
        if Path(path).exists():
            for i, l in json.loads(Path(path).read_text(encoding="utf-8")).items():
                if l.get("verdict"):
                    labels[i] = l
    return labels


def pct(a, b):
    return f"{100 * a / b:.0f}% ({a}/{b})" if b else "–"


def report_for(name: str, pairs: list[tuple[dict, str, str]]) -> list[str]:
    """pairs: (item, person_label, judge_label)."""
    lines = [f"\n### {name}\n"]
    n = len(pairs)
    said_real = [h for _, h, j in pairs if j == "real_error"]
    human_real = [j for _, h, j in pairs if h == "real_error"]
    lines.append(f"- agreement: {pct(sum(h == j for _, h, j in pairs), n)}")
    lines.append(f"- **real_error precision: {pct(sum(h == 'real_error' for h in said_real), len(said_real))}**")
    lines.append(f"- real_error recall: {pct(sum(j == 'real_error' for j in human_real), len(human_real))}")
    lines.append(f"- judge said uncertain: {pct(sum(j == 'uncertain' for _, _, j in pairs), n)}")

    lines.append("\nConfusion (rows = person, columns = judge):\n")
    lines.append("| person \\ judge | " + " | ".join(LABELS) + " |")
    lines.append("|---" * (len(LABELS) + 1) + "|")
    conf = Counter((h, j) for _, h, j in pairs)
    for h in LABELS:
        lines.append(f"| {h} | " + " | ".join(str(conf[(h, j)]) for j in LABELS) + " |")

    by_stratum = defaultdict(list)
    for it, h, j in pairs:
        by_stratum[it["stratum"]].append(h == j)
    lines.append("\nAgreement by stratum: " + ", ".join(
        f"{s} {pct(sum(v), len(v))}" for s, v in sorted(by_stratum.items())))
    return lines


def filter_report(round1_path: Path, diffs_path: Path, labels: dict) -> list[str]:
    """How the person labelled the round-1 items that build_diffs.py now
    filters out or splits -- a check that the filters only remove items
    that shouldn't become corrections anyway."""
    if not round1_path.exists() or not diffs_path.exists():
        return []
    current = {}
    for line in diffs_path.open(encoding="utf-8"):
        d = json.loads(line)
        current[d["id"]] = d["stratum"]
    groups = defaultdict(Counter)
    for line in round1_path.open(encoding="utf-8"):
        it = json.loads(line)
        if it["id"] not in labels:
            continue
        status = current.get(it["id"], "split")
        group = status if status in ("style", "filler", "garbled", "split") else "still judged"
        groups[group][to_label(it, labels[it["id"]]["verdict"])] += 1
    lines = ["\n## What the filters removed from round 1\n",
             "| round-1 items now | n | person's labels |", "|---|---|---|"]
    for g, c in sorted(groups.items(), key=lambda x: -sum(x[1].values())):
        lines.append(f"| {g} | {sum(c.values())} | " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + " |")
    return lines


def main():
    args = parse_args()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items, encoding="utf-8")}
    labels = merge_labels(args.labels.split(","))
    human = {i: to_label(items[i], l["verdict"]) for i, l in labels.items() if i in items}
    print(f"{len(human)} of {len(items)} current calibration items are hand-labelled")

    models = args.models.split(",")
    lines = [f"# Judge calibration ({len(human)} hand-labelled differences)\n",
             "Person's labels: " + ", ".join(f"{k} {v}" for k, v in Counter(human.values()).most_common())]
    lines += filter_report(Path(args.round1_items), Path(args.diffs), labels)

    for version in args.prompt_versions.split(","):
        verdicts = {m: load_cache(cache_path(Path(args.judge_dir), m), version) for m in models}
        if not any(verdicts.values()):
            continue
        lines.append(f"\n## Prompt {version}\n")
        for m in models:
            pairs = [(items[i], h, to_label(items[i], verdicts[m][i]["verdict"]))
                     for i, h in human.items() if i in verdicts[m]]
            if pairs:
                lines += report_for(m, pairs)
        unanimous = []
        for i, h in human.items():
            if all(i in verdicts[m] for m in models):
                js = {to_label(items[i], verdicts[m][i]["verdict"]) for m in models}
                unanimous.append((items[i], h, js.pop() if len(js) == 1 else "uncertain"))
        if unanimous:
            lines += report_for("all judges agree (disagreement = uncertain)", unanimous)

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(line for line in lines if not line.startswith("|")))


if __name__ == "__main__":
    main()
