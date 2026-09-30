#!/usr/bin/env python3
"""Cross-check the verifier against TypeSafe's Jev, an independent yes/no judge.

Jev (https://docs.typesafe.ai/api) answers a yes/no question about a piece of
state with a calibrated probability. It is a different model family from the
verifier and from the MiMo/DeepSeek judges that produced the verifier's
training labels, so agreement with it is evidence that doesn't share their
biases. It is NOT ground truth, so it is first checked against the hand
labels; only if it agrees with the person is it used on the pipeline's edits.

  calibration   the verifier's 98 hand-labelled examples (datasets/verifier-v2):
                AUC and precision/recall of Jev vs the person, next to the
                verifier's own -- "is Jev a judge worth trusting here?"
  applied       a score-stratified sample of candidates from correct.py's
                --keep-intermediates output: the share Jev says YES per
                verifier-score bin -- "does Jev agree the verifier's accepted
                edits are real fixes?"

Needs TYPESAFE_API_KEY in .env. Responses are cached per item, so reruns
cost nothing.

  python3 src/edit/jev_check.py --corrected outputs/edit/e2e/corrected-full.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

URL = "https://api.typesafe.ai/v1/systemone"
HUB_REPO = "PedramR/ASR_Post-processing"
QUESTION = (
    "A speech recognizer transcribed a Persian customer-service phone call (Digikala, an online store). "
    "`context` is the transcript around one spot, with the recognizer's words there marked by ⟦ ⟧. "
    "A correction is proposed: replace `original` with `proposed`. "
    "Is this a correct fix: the recognizer clearly misheard and `proposed` is what was actually said "
    "(a word that doesn't fit the sentence, a misheard name, brand or product)? "
    "Answer false if the original words are fine, if both are acceptable (spelling, half-space, "
    "colloquial vs formal, filler words), or if you cannot tell."
)


def state_of(item: dict) -> dict:
    original = item["original"] or "∅"
    state = {"context": f"{item['left']} ⟦{original}⟧ {item['right']}".strip(),
             "original": original, "proposed": item["proposed"]}
    if item.get("crm_names"):
        state["known_names"] = "، ".join(item["crm_names"])
    return state


def ask(item: dict, key: str, retries: int = 6) -> tuple[float | None, dict]:
    body = {"model": "jev-latest", "state": state_of(item),
            "questions": {"fix": {"type": "noul", "instructions": QUESTION}}}
    req = urllib.request.Request(URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            return out["answers"]["fix"]["noul"], out.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, {"error": e.code}
        except Exception as e:  # network hiccup: retry, then give up on this item
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, {"error": repr(e)}
    return None, {"error": "retries"}


class Cache:
    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                r = json.loads(line)
                self.data[r["k"]] = r["p"]
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(item: dict) -> str:
        return hashlib.sha1(json.dumps(state_of(item), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]

    def put(self, k: str, p: float):
        self.data[k] = p
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "p": p}) + "\n")


def jev_scores(items: list[dict], key: str, cache: Cache, workers: int) -> tuple[list[float | None], dict]:
    todo = {}
    for it in items:
        k = Cache.key(it)
        if k not in cache.data:
            todo[k] = it
    usage = {"input_tokens": 0, "output_tokens": 0, "failed": 0, "calls": len(todo)}
    with ThreadPoolExecutor(workers) as ex:
        for (k, _), (p, u) in zip(todo.items(), ex.map(lambda it: ask(it, key), todo.values())):
            if p is None:
                usage["failed"] += 1
                continue
            cache.put(k, p)
            usage["input_tokens"] += u.get("input_tokens", 0)
            usage["output_tokens"] += u.get("output_tokens", 0)
    return [cache.data.get(Cache.key(it)) for it in items], usage


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def prf(scores: list[float], labels: list[bool], t: float) -> dict:
    acc = [s >= t for s in scores]
    tp = sum(a and l for a, l in zip(acc, labels))
    return {"precision": tp / sum(acc) if sum(acc) else None, "recall": tp / sum(labels) if sum(labels) else None,
            "accepted": sum(acc)}


def fmt(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrected", default=None, help="correct.py output with --keep-intermediates (for the applied check).")
    p.add_argument("--per-bin", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="outputs/edit/jev/cache.jsonl")
    p.add_argument("--output", default="outputs/edit/jev/report.md")
    return p.parse_args()


def main():
    args = parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY is not set (put it in .env)")
    cache = Cache(Path(args.cache))
    lines = ["# Verifier vs Jev (TypeSafe)\n"]

    # 1. Is Jev a judge worth trusting? Check it against the person.
    cal_path = hf_hub_download(HUB_REPO, "verifier-qwen3.5-2b-v2/calibration_predictions.jsonl")
    cal = [json.loads(l) for l in open(cal_path, encoding="utf-8")]
    cal = [r for r in cal if r["label_source"] == "hand label"]
    jev, usage = jev_scores(cal, key, cache, args.workers)
    ok = [(r, j) for r, j in zip(cal, jev) if j is not None]
    labels = [r["label"] == "YES" for r, _ in ok]
    lines += [f"## Against the person's labels ({len(ok)} hand-labelled examples, {sum(labels)} YES)\n",
              f"{usage['calls']} new Jev calls, {usage['failed']} failed.\n",
              "| judge | AUC | threshold | accepted | precision | recall |", "|---|---|---|---|---|---|"]
    # Jev's probabilities sit low and compressed (YES items mostly 0.2-0.4), so
    # it gets its own thresholds; only ranking (AUC) is comparable across the two.
    for name, sc, ths in (("verifier", [r["p_yes"] for r, _ in ok], (0.5, 0.7, 0.8, 0.9)),
                          ("Jev", [j for _, j in ok], (0.2, 0.3, 0.4, 0.5))):
        for t in ths:
            m = prf(sc, labels, t)
            lines.append(f"| {name} | {fmt(auc(sc, labels), 3)} | {t} | {m['accepted']} | "
                         f"{fmt(m['precision'])} | {fmt(m['recall'])} |")

    # 2. Does Jev agree with the verifier's accepted edits?
    if args.corrected:
        rng = random.Random(args.seed)
        cands = [c for line in open(args.corrected, encoding="utf-8") for c in json.loads(line)["candidates"]]
        bins = [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]
        lines += ["\n## Jev on the pipeline's own candidates (main test set, score-stratified sample)\n",
                  "| verifier score | candidates | sampled | Jev mean | Jev >= 0.2 | Jev >= 0.3 |",
                  "|---|---|---|---|---|---|"]
        total_new = 0
        for lo, hi in bins:
            pool = [c for c in cands if lo <= c["score"] < hi]
            sample = rng.sample(pool, min(args.per_bin, len(pool)))
            js, u = jev_scores(sample, key, cache, args.workers)
            total_new += u["calls"]
            js = [j for j in js if j is not None]
            n = len(js)
            lines.append(f"| {lo:.1f}-{min(hi, 1.0):.1f} | {len(pool)} | {n} | {fmt(sum(js) / n if n else None)} | "
                         f"{fmt(sum(j >= 0.2 for j in js) / n if n else None)} | "
                         f"{fmt(sum(j >= 0.3 for j in js) / n if n else None)} |")
        lines.append(f"\n{total_new} new Jev calls for this table.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
