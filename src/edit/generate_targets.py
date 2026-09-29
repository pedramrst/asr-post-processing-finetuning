#!/usr/bin/env python3
"""Step 2e: generate the edit-method training set -- ~50-word Whisper
windows with verified correction targets.

Three stages, each resumable:

  select    Cut every row of build_diffs.py's output into windows of random
            length (--window-words-min..max; never cutting through a
            difference) and keep:
              * "judged" windows: 1-3 substitution differences per 50 words to judge
                (strata entity / phonetic_sub / other_sub), entity windows
                first;
              * "free" windows: nothing to judge, so the target is the
                Whisper text itself (a no-edit example for free).
            Windows with garbled or long differences are skipped, and at
            most --max-per-call windows come from one call. Insertions and
            deletions (whisper_dropped / whisper_extra) are out of scope for
            v1: they stay as Whisper has them. Saved to windows.jsonl and
            judge_items.jsonl; reused on later runs.
  judge     Cheap-first cascade with the calibrated rule (README step 2):
            MiMo@think on every difference, then DeepSeek@think only where
            MiMo didn't say either/soniox_wrong -- elsewhere DeepSeek's
            answer couldn't change the outcome. Both run in one shared pool
            (a difference goes to DeepSeek as soon as MiMo has answered it).
            Every answer is appended to the llm_judge.py cache as it arrives.
  assemble  An edit is accepted when DeepSeek says real_error and MiMo
            didn't reject it; a hand label (calibration rounds) overrides
            the judges. Windows with an undecided difference are left out
            (see decide). Writes train.jsonl (source/target/edits per
            window plus read-only left/right context, with a by-call
            train/validation split) and summary.json.

Progress, interrupting and resuming:
  python3 src/edit/generate_targets.py start    # run in the background (log: <out>/run.log)
  python3 src/edit/generate_targets.py status   # stage, counts, time left, cost so far + projected
  python3 src/edit/generate_targets.py stop     # graceful stop: in-flight calls are saved
  python3 src/edit/generate_targets.py run      # same as start, in the foreground (Ctrl+C to stop)
  python3 src/edit/generate_targets.py assemble # redo only the last stage from cached answers (no API calls)
Running start/run again resumes: cached answers are never paid for twice.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from calibrate import merge_labels
from edits import Edit, apply_edits, changes_number
from llm_judge import JudgeCache, cache_path, load_cache, make_client, run_tasks, to_label

load_dotenv()

CHEAP_JUDGE = "xiaomi/mimo-v2.6-flash@think"
MAIN_JUDGE = "deepseek/deepseek-v4.1-flash@think"
CHEAP_REJECTS = {"either_fine", "soniox_wrong"}
SUBSTITUTION_STRATA = {"entity", "phonetic_sub", "other_sub"}
SKIP_WINDOW_STRATA = {"garbled", "long"}
MAX_JUDGED_PER_WINDOW = 3
MIN_LAST_WINDOW_WORDS = 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "start", "status", "stop", "assemble"])
    p.add_argument("--targets-dir", default="outputs/edit/targets", help="build_diffs.py output (rows/diffs).")
    p.add_argument("--output-dir", default="outputs/edit/data")
    p.add_argument("--judge-dir", default="outputs/edit/judge")
    p.add_argument("--labels", default="outputs/edit/targets/calibration_labels.json,"
                   "outputs/edit/targets/calibration_labels_review.json")
    p.add_argument("--windows", type=int, default=2000)
    p.add_argument("--judged-frac", type=float, default=0.7, help="Share of windows with differences to judge.")
    p.add_argument("--window-words-min", type=int, default=40)
    p.add_argument("--window-words-max", type=int, default=80,
                   help="Each window's length is drawn uniformly from [min, max] (the first run used 50/50).")
    p.add_argument("--context-words", type=int, default=15,
                   help="Read-only context words kept on each side of a window (not edited, not judged).")
    p.add_argument("--max-per-call", type=int, default=2)
    p.add_argument("--validation-frac", type=float, default=0.05, help="Share of calls held out for validation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--think-max-tokens", type=int, default=8000,
                   help="DeepSeek often reasons past 4k tokens: at 4000, 35% of a test run hit the cap vs ~11% at 8000.")
    p.add_argument("--reselect", action="store_true", help="Redo the select stage even if windows.jsonl exists.")
    return p.parse_args()


# --------------------------------------------------------------------------- progress

class Progress:
    """progress.json: what `status` reads. Written atomically, at most once a second."""

    def __init__(self, path: Path):
        self.path, self.state, self._last = path, {}, 0.0

    def update(self, force: bool = False, **kw):
        self.state.update(kw, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if force or time.time() - self._last >= 1:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))
            tmp.replace(self.path)
            self._last = time.time()


# --------------------------------------------------------------------------- select

def cut_windows(row: dict, diffs: list[dict], min_words: int, max_words: int,
                rng: random.Random) -> list[tuple[int, int, list[dict]]]:
    """Consecutive [start, end) windows over the row's Whisper words, each of
    a random length in [min_words, max_words], skipping any window that a
    difference crosses; each with the differences inside it."""
    n = row["n_words"]
    out = []
    s = 0
    while s < n:
        e = min(s + rng.randint(min_words, max_words), n)
        if e - s < MIN_LAST_WINDOW_WORDS and s > 0:
            break
        if not any(d["start"] < b < d["end"] for d in diffs for b in (s, e)):  # nothing straddles a boundary
            inside = [d for d in diffs if s <= d["start"] and d["end"] <= e
                      and not (d["start"] == d["end"] == e and e < n)]  # an insertion at e belongs to the next window
            out.append((s, e, inside))
        s = e
    return out


def max_judged(window_words: int) -> int:
    """At most MAX_JUDGED_PER_WINDOW differences per 50 words."""
    return max(1, round(MAX_JUDGED_PER_WINDOW * window_words / 50))


def select(args, out: Path) -> tuple[list[dict], list[dict]]:
    rows = {}
    for line in open(Path(args.targets_dir) / "rows.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[r["row_id"]] = r
    by_row = defaultdict(list)
    for line in open(Path(args.targets_dir) / "diffs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        by_row[d["row_id"]].append(d)

    judged, free = [], []
    for row_id, row in rows.items():
        rng_row = random.Random(f"{args.seed}:{row_id}")
        for s, e, inside in cut_windows(row, by_row[row_id], args.window_words_min, args.window_words_max, rng_row):
            if any(d["stratum"] in SKIP_WINDOW_STRATA for d in inside):
                continue
            # kind check too: an insertion whose Soniox side is a CRM name is
            # stratum "entity", but insertions are out of scope for v1.
            to_judge = [d for d in inside if d["stratum"] in SUBSTITUTION_STRATA and d["kind"] == "substitute"]
            w = {"window_id": f"{row_id}:{s}", "row_id": row_id, "call_id": row["call_id"], "start": s, "end": e,
                 "crm_names": row["crm_names"], "judge_ids": [d["id"] for d in to_judge],
                 "has_entity": any(d["stratum"] == "entity" for d in to_judge), "_diffs": to_judge}
            if not to_judge:
                free.append(w)
            elif len(to_judge) <= max_judged(e - s):
                judged.append(w)

    rng = random.Random(args.seed)
    rng.shuffle(judged)
    rng.shuffle(free)
    judged.sort(key=lambda w: not w["has_entity"])  # stable: entity windows first, random within each group
    per_call = Counter()
    picked = []
    for pool, quota in ((judged, round(args.windows * args.judged_frac)),
                        (free, args.windows - round(args.windows * args.judged_frac))):
        n = 0
        for w in pool:
            if n == quota:
                break
            if per_call[w["call_id"]] < args.max_per_call:
                per_call[w["call_id"]] += 1
                picked.append(w)
                n += 1
        if n < quota:
            print(f"  only {n}/{quota} windows available of this kind -- build_diffs.py on more rows gives more")

    items = [d for w in picked for d in w.pop("_diffs")]
    for w in judged + free:
        w.pop("_diffs", None)
    for name, data in (("windows.jsonl", picked), ("judge_items.jsonl", items)):
        with open(out / name, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"Selected {len(picked)} windows ({sum(bool(w['judge_ids']) for w in picked)} to judge, "
          f"{sum(not w['judge_ids'] for w in picked)} free), {len(items)} differences to judge")
    return picked, items


# --------------------------------------------------------------------------- judge

def judge(args, items: list[dict], progress: Progress) -> dict:
    """Cheap-first cascade, per difference: MiMo answers first and, unless it
    rules the edit out, DeepSeek is asked right away -- one shared pool of
    --workers parallel calls, so the two models' calls overlap instead of
    running as two passes one after the other."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY must be set in .env")
    client = make_client(key)
    judge_dir = Path(args.judge_dir)
    judge_dir.mkdir(parents=True, exist_ok=True)
    cheap = JudgeCache(judge_dir, CHEAP_JUDGE, args.think_max_tokens)
    main = JudgeCache(judge_dir, MAIN_JUDGE, args.think_max_tokens)

    def settled(it) -> bool:
        return cheap.has(it["id"]) and (cheap.label(it) in CHEAP_REJECTS or main.has(it["id"]))

    def cascade(it) -> float:
        """Judges one difference; returns what its new calls cost."""
        spent = 0.0
        if not cheap.has(it["id"]):
            spent += cheap.judge(client, it).get("cost") or 0
        if cheap.label(it) not in CHEAP_REJECTS and not main.has(it["id"]):
            spent += main.judge(client, it).get("cost") or 0
        return spent

    def stats() -> dict:
        asked_main = [it for it in items if cheap.has(it["id"]) and cheap.label(it) not in CHEAP_REJECTS]
        return {CHEAP_JUDGE: cheap.stats(items), MAIN_JUDGE: main.stats(asked_main)}

    todo = [it for it in items if not settled(it)]
    counters = {"done": len(items) - len(todo), "new": 0, "started": time.time(), "last_cost": 0.0, "spent": 0.0}
    print(f"{len(items) - len(todo)} differences already settled, {len(todo)} to judge", flush=True)

    def on_done(spent):
        counters["done"] += 1
        counters["new"] += 1
        counters["spent"] += spent or 0
        rate = counters["new"] / max(time.time() - counters["started"], 1e-6)
        left = len(items) - counters["done"]
        extra = {}
        if time.time() - counters["last_cost"] > 5:  # the breakdown scans all items: not on every result
            counters["last_cost"] = time.time()
            breakdown = stats()
            extra = {"cost_by_model_usd": {m: v["cost_usd"] for m, v in breakdown.items()},
                     "cost_total_usd": round(sum(v["cost_usd"] for v in breakdown.values()), 4)}
        progress.update(stage="judge", done=counters["done"], total=len(items),
                        eta_minutes=round(left / rate / 60, 1),
                        cost_this_run_usd=round(counters["spent"], 4),
                        # average cost of this run's differences so far, times the ones left
                        cost_projected_this_run_usd=round(counters["spent"] * (1 + left / counters["new"]), 2),
                        **extra)

    progress.update(force=True, stage="judge", done=counters["done"], total=len(items), eta_minutes=None)
    try:
        run_tasks([lambda it=it: cascade(it) for it in todo], args.workers, on_done, desc="judging")
    finally:
        cheap.close()
        main.close()
    result = stats()
    progress.update(force=True, judge=result, cost_this_run_usd=round(counters["spent"], 4),
                    cost_by_model_usd={m: v["cost_usd"] for m, v in result.items()},
                    cost_total_usd=round(sum(v["cost_usd"] for v in result.values()), 4))
    print(json.dumps(result, indent=2), flush=True)
    return result


# --------------------------------------------------------------------------- assemble

CONFIDENT_REJECTS = {"either_fine", "soniox_wrong"}


def decide(item: dict, cheap: dict, main: dict, human: dict) -> tuple[str, str]:
    """-> ("accepted" | "rejected" | "undecided", reason).

    accepted:  DeepSeek says real_error and MiMo didn't reject it (the
               calibrated rule), or a hand label says real_error.
    rejected:  a judge (or hand label) confidently says there's nothing to
               fix -- either version is fine, or Whisper was right.
    undecided: anything else -- DeepSeek ran out of tokens, a judge was
               unsure, or both versions are wrong. There may be a real error
               here, so the window can't serve as a "leave this alone"
               example either (see assemble)."""
    if item["id"] in human:
        label = to_label(item, human[item["id"]]["verdict"])
        status = ("accepted" if label == "real_error" else
                  "rejected" if label in CONFIDENT_REJECTS else "undecided")
        return status, "hand label"
    c = to_label(item, cheap[item["id"]]["verdict"]) if item["id"] in cheap else "missing"
    m = to_label(item, main[item["id"]]["verdict"]) if item["id"] in main else "missing"
    why = f"judges ({c} / {m})"
    if m == "real_error" and c not in CHEAP_REJECTS:
        return "accepted", why
    if c in CONFIDENT_REJECTS or m in CONFIDENT_REJECTS:
        return "rejected", why
    return "undecided", why


def assemble(args, out: Path, windows: list[dict], items: list[dict], judge_stats: dict) -> dict:
    rows = {}
    for line in open(Path(args.targets_dir) / "rows.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[r["row_id"]] = r
    by_id = {it["id"]: it for it in items}
    judge_dir = Path(args.judge_dir)
    cheap = load_cache(cache_path(judge_dir, CHEAP_JUDGE))
    main = load_cache(cache_path(judge_dir, MAIN_JUDGE))
    human = merge_labels(args.labels.split(","))

    calls = sorted({w["call_id"] for w in windows})
    random.Random(args.seed).shuffle(calls)
    validation = set(calls[:round(len(calls) * args.validation_frac)])

    counts = Counter()
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for w in windows:
            row_words = rows[w["row_id"]]["whisper"].split()
            source = " ".join(row_words[w["start"]:w["end"]])
            edits, undecided = [], 0
            for diff_id in w["judge_ids"]:
                it = by_id[diff_id]
                status, why = decide(it, cheap, main, human)
                counts["judged"] += 1
                counts[status] += 1
                counts["by hand label"] += why == "hand label"
                undecided += status == "undecided"
                if status == "accepted":
                    # A hard safety rule, applied here regardless of build_diffs.py's
                    # stratum (which predates this rule for windows already selected):
                    # never train on an edit that changes a number -- see edits.py's
                    # touches_number and README's "Never changes numbers".
                    if changes_number(it["whisper_span"].split(), it["soniox_span"].split()):
                        counts["accepted but excluded (touches a number)"] += 1
                    else:
                        edits.append(Edit(it["start"] - w["start"], it["end"] - w["start"],
                                          it["whisper_span"].split(), it["soniox_span"].split(),
                                          meta={"diff_id": diff_id, "stratum": it["stratum"], "decided_by": why}))
            if undecided:
                # Might hold a real error the target would leave in: training on
                # it would teach the model to skip real corrections.
                counts["windows excluded (undecided difference)"] += 1
                continue
            target = apply_edits(source, edits)
            counts["windows with edits" if edits else "windows without edits"] += 1
            split = "validation" if w["call_id"] in validation else "train"
            counts[f"{split} windows"] += 1
            f.write(json.dumps({
                "window_id": w["window_id"], "call_id": w["call_id"], "split": split,
                # Read-only context from the same row (Whisper text, after rules):
                # shown to the model but never edited, so edits near a window's
                # edge still have surrounding words -- as they will at inference,
                # where a long transcript is cut into windows the same way.
                "left_context": " ".join(row_words[max(0, w["start"] - args.context_words):w["start"]]),
                "source": source,
                "right_context": " ".join(row_words[w["end"]:w["end"] + args.context_words]),
                "target": target, "crm_names": w["crm_names"],
                "edits": [{"start": e.start, "end": e.end, "original": " ".join(e.original),
                           "replacement": " ".join(e.replacement), **e.meta} for e in edits],
            }, ensure_ascii=False) + "\n")
    summary = {**counts, "judge": judge_stats,
               "cost_usd": round(sum(s["cost_usd"] for s in judge_stats.values()), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# --------------------------------------------------------------------------- commands

def _stop_on_sigterm(signum, frame):
    raise KeyboardInterrupt


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, _stop_on_sigterm)  # `stop` behaves like Ctrl+C
    progress = Progress(out / "progress.json")
    progress.update(force=True, state="running", stage="select", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        if (out / "windows.jsonl").exists() and not args.reselect:
            windows = [json.loads(l) for l in open(out / "windows.jsonl", encoding="utf-8")]
            items = [json.loads(l) for l in open(out / "judge_items.jsonl", encoding="utf-8")]
            print(f"Reusing {len(windows)} selected windows, {len(items)} differences (--reselect to redo)")
        else:
            windows, items = select(args, out)
        progress.update(force=True, windows=len(windows), differences_to_judge=len(items))
        stats = judge(args, items, progress)
        progress.update(force=True, stage="assemble")
        summary = assemble(args, out, windows, items, stats)
        progress.update(force=True, state="finished", stage="done", summary=summary)
    except KeyboardInterrupt:
        progress.update(force=True, state="stopped (resume with start or run)")
        print("Stopped. Everything judged so far is cached; run again to resume.", flush=True)
        (out / "run.pid").unlink(missing_ok=True)
        # Results are already flushed; don't wait for a hung HTTP call's worker thread.
        os._exit(130)
    except Exception as e:
        progress.update(force=True, state=f"failed: {type(e).__name__}: {e}")
        raise
    finally:
        (out / "run.pid").unlink(missing_ok=True)


def running_pid(out: Path) -> int | None:
    pid_file = out / "run.pid"
    if not pid_file.exists():
        return None
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def main():
    args = parse_args()
    out = Path(args.output_dir)
    if args.command == "run":
        return run(args)
    if args.command == "assemble":  # redo only the last stage, from cached judge answers (no API calls)
        windows = [json.loads(l) for l in open(out / "windows.jsonl", encoding="utf-8")]
        items = [json.loads(l) for l in open(out / "judge_items.jsonl", encoding="utf-8")]
        previous = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        return assemble(args, out, windows, items, previous.get("judge", {}))
    if args.command == "start":
        if running_pid(out):
            sys.exit(f"Already running (pid {running_pid(out)}); see `status`.")
        out.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-u", __file__, "run"] + [a for a in sys.argv[1:] if a != "start"]
        with open(out / "run.log", "a", encoding="utf-8") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started (pid {proc.pid}). Log: {out / 'run.log'}\n"
              f"  status: python3 src/edit/generate_targets.py status\n"
              f"  stop:   python3 src/edit/generate_targets.py stop")
        return
    if args.command == "stop":
        pid = running_pid(out)
        if not pid:
            sys.exit("Not running.")
        os.kill(pid, signal.SIGTERM)
        print(f"Sent stop to pid {pid}; it saves the calls in flight, then exits (see `status`).")
        return
    # status
    state = json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else {}
    if not state:
        sys.exit("No run yet.")
    pid = running_pid(out)
    print(f"process: {'running, pid ' + str(pid) if pid else 'not running'}")
    if "cost_this_run_usd" in state:
        print(f"cost: ${state['cost_this_run_usd']:.2f} spent in this run"
              + (f", ~${state['cost_projected_this_run_usd']:.2f} projected by the end"
                 if state.get("cost_projected_this_run_usd") is not None and state.get("stage") == "judge" else "")
              + (f" | dataset total incl. reused answers ${state['cost_total_usd']:.2f}" if "cost_total_usd" in state else "")
              + (f" | by model: {state['cost_by_model_usd']}" if "cost_by_model_usd" in state else ""))
    if state.get("stage") == "judge" and state.get("total"):
        print(f"progress: {state['done']}/{state['total']} differences"
              f" ({100 * state['done'] / state['total']:.0f}%), ~{state.get('eta_minutes')} min left")
    for k, v in state.items():
        if k not in ("summary", "judge", "cost_by_model_usd"):
            print(f"  {k}: {v}")
    if state.get("summary"):
        print("  summary:", json.dumps(state["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
