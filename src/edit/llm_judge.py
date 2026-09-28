#!/usr/bin/env python3
"""Step 2b of the edit-based method: have cheap LLMs rule on each
Whisper-vs-Soniox difference from build_diffs.py.

Each difference is shown **blind**: the two versions of the marked span
appear as "A" and "B" in an order fixed per item by its id (ab_order), so
neither the judge nor a person labelling the same item (make_label_sheet.py
uses the same order) knows which one is Soniox. A judge that knew Soniox is
usually the better system would just learn to side with it -- the one
failure this step exists to catch.

Verdicts: A / B (that version is clearly right, the other clearly wrong),
either (both acceptable: spelling/dialect variants, small variants that
both fit, filler words), neither (both wrong; `correct` gives the right
text), unsure (can't tell without the audio). to_label() maps them back to
the target-cleaning labels:

    soniox chosen  -> real_error     (Whisper wrong: keep as a correction)
    whisper chosen -> soniox_wrong   (no edit)
    either         -> either_fine    (no edit)
    neither        -> both_wrong     (correction to `correct`, if trusted)
    unsure         -> uncertain      (no edit)

Calls go through OpenRouter (OPENROUTER_API_KEY in .env, as for
telegram_agent.py). Results are cached per model in
<output-dir>/<model>.jsonl, keyed by item id and PROMPT_VERSION, so a rerun
only calls the API for new or failed items.

Example:
  python3 src/edit/llm_judge.py --items outputs/edit/targets/calibration_items.jsonl
  python3 src/edit/llm_judge.py --items ... --models deepseek/deepseek-v4.1-flash@nothink --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# "<openrouter model>@think" lets the model reason before answering (with a
# large token budget -- DeepSeek's reasoning alone often exceeds 2k tokens
# and then returns an empty answer); "@nothink" turns reasoning off (about
# 10x cheaper and faster). Calibration decides which is worth it.
DEFAULT_MODELS = [
    "deepseek/deepseek-v4.1-flash@think", "deepseek/deepseek-v4.1-flash@nothink",
    "xiaomi/mimo-v2.6-flash@think", "xiaomi/mimo-v2.6-flash@nothink",
]
THINK_MAX_TOKENS = 8000
# Per-call HTTP timeout, and how long a stopped run waits for calls already
# in flight. The OpenAI client's defaults (600s, plus its own retries) let a
# single hung call block a graceful stop for many minutes.
CALL_TIMEOUT_S = 120
STOP_GRACE_S = 60
STOP = threading.Event()  # set when a run is interrupted
NOTHINK_MAX_TOKENS = 300
# v2 (after hand-labelling round 1): adds "either" and makes A/B strict. In
# round 1 the person used "same" for anything that doesn't matter -- filler
# words, small variants that both fit (خدانگهدارتون/خدانگهدار, الی/تا) --
# while the judges read "same" as spelling-only and forced an A/B pick.
PROMPT_VERSION = "v2"
VERDICTS = {"A", "B", "either", "neither", "unsure"}

SYSTEM_PROMPT = """\
You check speech-recognition transcripts of Persian customer-service phone calls (Digikala, an Iranian \
online store). Speakers use casual spoken Tehrani Persian.

Two automatic transcripts of the same audio differ in one marked span. You get the surrounding words and \
the two versions of the span, labelled A and B (the order is random). The goal is to find real \
recognition errors -- places where one version is clearly wrong -- not to pick a preferred wording.

Answer:
- "A" or "B" only if that version is clearly what was said AND the other is clearly wrong: a different \
word that doesn't fit the sentence, a misheard name, a word that isn't Persian, or missing/extra words \
that break the sentence. Recognition errors usually SOUND similar to what was said.
- "either" if both versions are acceptable: the same words in a different spelling, spacing/half-space, \
or colloquial vs formal form; small variants that both fit (a different suffix, تا vs الی); or filler \
words (بله، آره، خب، الو، و، رو، ام) present in one version and not the other.
- "neither" if both are clearly wrong and you are confident what was said; put it in "correct".
- "unsure" if one version might be wrong but you can't tell which without the audio.

Notes:
- Colloquial forms are correct when that's how people speak (میشه، می‌خوام، بفرمایین، رو). Never prefer a \
version just because it is more formal.
- "Known names" come from the store's customer record for this call. Treat them as the correct spelling \
of those people's names, but only when that person is actually being mentioned.
- A span shown as ∅ means that version has no words there.
- The context is itself automatic output and may contain other errors; judge only the marked span.

Reply with only a JSON object: {"verdict": "A" | "B" | "either" | "neither" | "unsure", \
"correct": "<the right span, only for neither>", "confidence": "high" | "medium" | "low"}"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", required=True, help="Difference items .jsonl (from build_diffs.py / make_label_sheet.py).")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--output-dir", default="outputs/edit/judge")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def ab_order(item: dict) -> str:
    """"WS" = Whisper shown as A, "SW" = Soniox shown as A; fixed by the id."""
    return "WS" if int(item["id"], 16) % 2 == 0 else "SW"


def ab_spans(item: dict) -> tuple[str, str]:
    w, s = item["whisper_span"] or "∅", item["soniox_span"] or "∅"
    return (w, s) if ab_order(item) == "WS" else (s, w)


def user_prompt(item: dict) -> str:
    a, b = ab_spans(item)
    names = "، ".join(item.get("crm_names") or []) or "(none)"
    return (f"Known names: {names}\n\n"
            f"Context: {item['left']} ⟦ … ⟧ {item['right']}\n\n"
            f"A: {a}\nB: {b}")


def to_label(item: dict, verdict: str) -> str:
    """Blind verdict -> target-cleaning label. "same" is round 1's (v1)
    verdict, which the person used to mean what "either" means now."""
    if verdict in ("A", "B"):
        whisper_letter = "A" if ab_order(item) == "WS" else "B"
        return "soniox_wrong" if verdict == whisper_letter else "real_error"
    return {"either": "either_fine", "same": "either_fine", "neither": "both_wrong",
            "unsure": "uncertain"}.get(verdict, "invalid")


def parse_reply(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        d = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return d if d.get("verdict") in VERDICTS else None


def make_client(api_key: str) -> OpenAI:
    """OpenRouter client with a bounded per-call timeout and no client-side
    retries (judge_one retries itself)."""
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key,
                  timeout=CALL_TIMEOUT_S, max_retries=0)


def split_spec(spec: str) -> tuple[str, bool]:
    """"model@think" / "model@nothink" -> (model, think)."""
    model, _, mode = spec.partition("@")
    if mode not in ("think", "nothink"):
        raise ValueError(f"model spec needs @think or @nothink: {spec!r}")
    return model, mode == "think"


def judge_one(client: OpenAI, spec: str, item: dict, retries: int = 4, max_tokens: int | None = None) -> dict:
    model, think = split_spec(spec)
    max_tokens = max_tokens or (THINK_MAX_TOKENS if think else NOTHINK_MAX_TOKENS)
    extra = {"usage": {"include": True}}
    if not think:
        extra["reasoning"] = {"enabled": False}
    last_error = None
    for attempt in range(retries):
        if STOP.is_set():
            return {"id": item["id"], "prompt_version": PROMPT_VERSION, "model": spec,
                    "error": "stopped before answering"}  # not final: judged again on resume
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt(item)}],
                temperature=0,
                max_tokens=max_tokens,
                extra_body=extra,
            )
            text = resp.choices[0].message.content or ""
            parsed = parse_reply(text)
            usage = resp.usage.model_dump() if resp.usage else {}
            if parsed:
                return {"id": item["id"], "prompt_version": PROMPT_VERSION, "model": spec,
                        "verdict": parsed["verdict"], "label": to_label(item, parsed["verdict"]),
                        "correct": parsed.get("correct") or None, "confidence": parsed.get("confidence"),
                        "cost": usage.get("cost"), "raw": text}
            last_error = f"unparseable reply: {text[:200]!r}"
            if resp.choices[0].finish_reason == "length":
                # Deterministic at temperature 0 -- a retry (or a resumed run)
                # would hit the same limit, so this is recorded as final.
                return {"id": item["id"], "prompt_version": PROMPT_VERSION, "model": spec,
                        "error": "hit max_tokens before answering", "max_tokens": max_tokens,
                        "cost": usage.get("cost")}
        except Exception as e:  # network/rate-limit/provider errors: retry with backoff
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(2 ** attempt)
    return {"id": item["id"], "prompt_version": PROMPT_VERSION, "model": spec, "error": last_error}


def load_cache(path: Path, prompt_version: str = PROMPT_VERSION) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("prompt_version") == prompt_version and "verdict" in r:
            done[r["id"]] = r
    return done


def load_final_failures(path: Path, max_tokens: int) -> dict[str, dict]:
    """Items that already ran out of tokens at >= max_tokens: rerunning them
    would only fail again, so a resumed run skips them."""
    if not path.exists():
        return {}
    failed = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if (r.get("prompt_version") == PROMPT_VERSION and r.get("error") == "hit max_tokens before answering"
                and (r.get("max_tokens") or 0) >= max_tokens):
            failed[r["id"]] = r
    return failed


def cache_path(out_dir: Path, spec: str) -> Path:
    return out_dir / f"{spec.replace('/', '__')}.jsonl"


class JudgeCache:
    """One model's answers: the <model>.jsonl cache file, loaded once and
    appended to (thread-safely, flushed per line) as answers arrive, so an
    interrupted run loses nothing it paid for."""

    def __init__(self, out_dir: Path, spec: str, max_tokens: int | None = None):
        self.spec = spec
        self.path = cache_path(out_dir, spec)
        self.max_tokens = max_tokens or (THINK_MAX_TOKENS if split_spec(spec)[1] else NOTHINK_MAX_TOKENS)
        self.done = load_cache(self.path)
        self.failed = load_final_failures(self.path, self.max_tokens)
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8")

    def has(self, item_id: str) -> bool:
        """Answered, or already failed for good (would only fail again)."""
        return item_id in self.done or item_id in self.failed

    def label(self, item: dict) -> str:
        r = self.done.get(item["id"])
        return to_label(item, r["verdict"]) if r else "missing"

    def judge(self, client: OpenAI, item: dict) -> dict:
        """Cached answer, or a new call whose result is saved immediately."""
        if item["id"] in self.done:
            return self.done[item["id"]]
        r = judge_one(client, self.spec, item, 4, self.max_tokens)
        with self._lock:
            self._file.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._file.flush()
            if "verdict" in r:
                self.done[item["id"]] = r
            elif r.get("max_tokens"):
                self.failed[item["id"]] = r
        return r

    def stats(self, items: list[dict]) -> dict:
        ids = {it["id"] for it in items}
        judged = ids & set(self.done)
        failed = ids & set(self.failed)
        cost = sum(self.done[i].get("cost") or 0 for i in judged) + sum(self.failed[i].get("cost") or 0 for i in failed)
        return {"judged": len(judged), "failed": len(failed), "errors": len(ids - judged - failed),
                "cost_usd": round(cost, 4)}

    def close(self):
        self._file.close()


def run_tasks(tasks: list, workers: int, on_done=None, desc: str = "judging"):
    """Runs zero-argument callables on a thread pool, calling on_done(result)
    as each finishes.

    Ctrl+C / SIGTERM (as KeyboardInterrupt in the main thread): pending tasks
    are cancelled, judge_one stops retrying (STOP), the tasks already in
    flight are waited for up to STOP_GRACE_S so their answers get saved, then
    KeyboardInterrupt is re-raised. A second interrupt stops immediately."""
    pool = ThreadPoolExecutor(workers)
    pending = {pool.submit(t) for t in tasks}
    bar = tqdm(total=len(pending), desc=desc)
    try:
        for fut in as_completed(pending):
            r = fut.result()
            bar.update(1)
            if on_done:
                on_done(r)
    except KeyboardInterrupt:
        STOP.set()
        for fut in pending:
            fut.cancel()
        in_flight = [fut for fut in pending if not fut.done() and not fut.cancelled()]
        print(f"\nInterrupted: saving {len(in_flight)} calls already in flight, for up to "
              f"{STOP_GRACE_S}s (interrupt again to stop immediately)...", flush=True)
        try:
            for fut in as_completed(in_flight, timeout=STOP_GRACE_S):
                bar.update(1)
                if on_done:
                    on_done(fut.result())
        except TimeoutError:
            print("Some calls didn't finish in time; they'll be judged again on resume.", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        raise KeyboardInterrupt
    finally:
        bar.close()
    pool.shutdown(wait=False)


def run_models(client: OpenAI, specs: list[str], items: list[dict], out_dir: Path, workers: int,
               max_tokens: int | None = None) -> dict[str, dict]:
    """Judges `items` with every model in `specs` side by side -- one shared
    pool, so the total time is about the slowest model's, not the sum."""
    caches = {s: JudgeCache(out_dir, s, max_tokens) for s in specs}
    tasks = []
    for s, cache in caches.items():
        todo = [it for it in items if not cache.has(it["id"])]
        print(f"{s}: {len(items) - len(todo)} cached or failed before, {len(todo)} to judge")
        tasks += [lambda c=cache, it=it: c.judge(client, it) for it in todo]
    try:
        run_tasks(tasks, workers)
    finally:
        for c in caches.values():
            c.close()
    return {s: c.stats(items) for s, c in caches.items()}


def main():
    args = parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY must be set in .env")
    client = make_client(key)
    items = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    if args.limit:
        items = items[:args.limit]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, stats in run_models(client, args.models.split(","), items, out_dir, args.workers).items():
        print(model, stats)


if __name__ == "__main__":
    main()
