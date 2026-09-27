#!/usr/bin/env python3
"""Step 2c: sample a calibration set of differences and build a local HTML
page to label them by hand.

The hand labels are the yardstick for the LLM judges (calibrate.py): before
trusting a judge on thousands of differences, measure how often it agrees
with a person on these. Items are drawn per stratum (build_diffs.py) so
names, sound-alike substitutions, dropped words, etc. are all represented,
with at most MAX_PER_CALL items from any one call.

The page is a single self-contained file opened straight from disk (no
server, nothing uploaded -- it embeds real call text). It shows each item
blind as A/B in the same order llm_judge.py uses, autosaves in the
browser, and exports the labels as JSON: save that file as
<output-dir>/calibration_labels.json.

Two more modes, for later labelling rounds:
  --refresh  re-map the existing calibration items onto a rebuilt
             diffs.jsonl (keeping labels of unchanged items)
  --review   a page with only the items the person and the judges
             disagree on, plus unlabelled pieces of split items

Example:
  python3 src/edit/make_label_sheet.py
  open outputs/edit/targets/label_calibration.html
  python3 src/edit/make_label_sheet.py --refresh
  python3 src/edit/make_label_sheet.py --review
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_diffs import NOT_JUDGED
from calibrate import merge_labels
from llm_judge import DEFAULT_MODELS, ab_spans, cache_path, load_cache, to_label

QUOTAS = {"entity": 50, "phonetic_sub": 45, "other_sub": 35, "whisper_dropped": 30,
          "whisper_extra": 25, "long": 5}
MAX_PER_CALL = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diffs", default="outputs/edit/targets/diffs.jsonl")
    p.add_argument("--output-dir", default="outputs/edit/targets")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--refresh", action="store_true",
                   help="Re-map the existing calibration items onto the current diffs.jsonl (see refresh()).")
    p.add_argument("--review", action="store_true",
                   help="Build label_review.html with only the items needing another look (see review()).")
    p.add_argument("--labels", default="outputs/edit/targets/calibration_labels.json,"
                   "outputs/edit/targets/calibration_labels_review.json",
                   help="Comma-separated label files for --review; later files override earlier ones.")
    p.add_argument("--judge-dir", default="outputs/edit/judge")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    return p.parse_args()


def sample(diffs: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for d in diffs:
        by_stratum[d["stratum"]].append(d)
    per_call = Counter()
    picked = []
    for stratum, quota in QUOTAS.items():
        pool = by_stratum[stratum][:]
        rng.shuffle(pool)
        n = 0
        for d in pool:
            if n == quota:
                break
            if per_call[d["call_id"]] < MAX_PER_CALL:
                per_call[d["call_id"]] += 1
                picked.append(d)
                n += 1
        if n < quota:
            print(f"  only {n}/{quota} available for stratum {stratum}")
    rng.shuffle(picked)  # don't present strata in blocks
    return picked


PAGE = """<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration Labels</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e2e0; --a:#1f6feb; --b:#b35900;
        --mark-a:#dbe9ff; --mark-b:#ffe7cc; --done:#1a7f37; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --card:#1f1f22; --ink:#ececee; --muted:#9a9aa0;
        --line:#333338; --a:#6ea8ff; --b:#ffb366; --mark-a:#1d3357; --mark-b:#4a3014; --done:#4ac26b; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.9 Vazirmatn, Tahoma, sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between; }
h1 { font-size:18px; margin:0; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:10px 0 16px; }
.bar > div { height:100%; background:var(--done); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
.meta { color:var(--muted); font-size:13px; direction:ltr; text-align:left; }
.names { color:var(--muted); font-size:14px; margin:4px 0 12px; }
.ver { padding:8px 10px; border-radius:8px; margin:6px 0; border:1px solid var(--line); }
.ver b { display:inline-block; min-width:1.6em; }
.ver.a b { color:var(--a); } .ver.b b { color:var(--b); }
.ver.a mark { background:var(--mark-a); color:inherit; padding:0 3px; border-radius:3px; }
.ver.b mark { background:var(--mark-b); color:inherit; padding:0 3px; border-radius:3px; }
.btns { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 8px; }
button { font:inherit; font-size:15px; padding:6px 12px; border-radius:8px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); cursor:pointer; }
button.on { outline:2px solid var(--done); }
input[type=text] { font:inherit; width:100%; padding:6px 10px; border:1px solid var(--line); border-radius:8px;
                   background:var(--bg); color:var(--ink); margin-top:6px; }
.nav { display:flex; flex-wrap:wrap; gap:8px; justify-content:space-between; margin-top:14px; }
.help { color:var(--muted); font-size:13px; margin-top:18px; }
.prev { color:var(--muted); font-size:14px; }
kbd { font-size:12px; border:1px solid var(--line); border-radius:4px; padding:0 4px; }
</style></head><body><main>
<header><h1>برچسب‌گذاری تفاوت‌ها</h1>
<div><span id="count"></span> &nbsp; <button id="export">Export labels</button>
<label><button onclick="document.getElementById('imp').click()">Import</button>
<input id="imp" type="file" accept=".json" hidden></label></div></header>
<div class="bar"><div id="prog"></div></div>
<div class="card">
  <div class="meta" id="meta"></div>
  <div class="names" id="names"></div>
  <div class="ver a" id="va"></div>
  <div class="ver b" id="vb"></div>
  <div class="btns" id="btns"></div>
  <div class="prev" id="prevlabel"></div>
  <input type="text" id="correct" placeholder="متن درست (فقط برای neither — فقط خود متن)">
  <input type="text" id="note" placeholder="یادداشت (اختیاری، هر توضیحی)">
  <div class="nav"><button id="prev">→ قبلی</button><button id="nextU">بعدیِ برچسب‌نخورده</button><button id="next">بعدی ←</button></div>
</div>
<div class="help">
<b>A</b> / <b>B</b>: that version is clearly right <i>and</i> the other clearly wrong &nbsp;·&nbsp;
<b>either</b>: both acceptable (same words spelled differently, small variants that both fit, filler words) &nbsp;·&nbsp;
<b>neither</b>: both clearly wrong (type the right text in the first box; comments go in the note box) &nbsp;·&nbsp;
<b>unsure</b>: one might be wrong but you can't tell which.
∅ = that version has no words there. "Known names" are from the customer record. Order of A/B is random.
Keys: <kbd>1</kbd> A <kbd>2</kbd> B <kbd>3</kbd> either <kbd>4</kbd> neither <kbd>5</kbd> unsure, <kbd>←</kbd>/<kbd>→</kbd> navigate.
Labels autosave in this browser; <b>Export</b> and save as <code>__EXPORT_NAME__</code> when done.
</div></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "__STORAGE_KEY__";
const EXPORT_NAME = "__EXPORT_NAME__";
const VERDICTS = [["A","A"],["B","B"],["either","either"],["neither","neither"],["unsure","unsure"]];
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
let i = 0;
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function esc(s) { return s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function line(it, span) { return esc(it.left) + " <mark>" + esc(span) + "</mark> " + esc(it.right); }
function render() {
  const it = ITEMS[i], l = labels[it.id] || {};
  $("meta").textContent = (i + 1) + " / " + ITEMS.length + "   ·   " + it.id;
  $("names").textContent = "نام‌های شناخته‌شده: " + (it.crm_names.length ? it.crm_names.join("، ") : "—");
  $("va").innerHTML = "<b>A</b> " + line(it, it.a);
  $("vb").innerHTML = "<b>B</b> " + line(it, it.b);
  $("btns").innerHTML = "";
  VERDICTS.forEach(([v, t], k) => {
    const b = document.createElement("button");
    b.textContent = (k + 1) + " · " + t; if (l.verdict === v) b.className = "on";
    b.onclick = () => setVerdict(v); $("btns").appendChild(b);
  });
  $("prevlabel").textContent = it.prev ? "Your earlier answer: " + it.prev : "";
  $("correct").value = l.correct || ""; $("note").value = l.note || "";
  const n = ITEMS.filter(x => labels[x.id] && labels[x.id].verdict).length;
  $("count").textContent = n + " / " + ITEMS.length + " labelled";
  $("prog").style.width = (100 * n / ITEMS.length) + "%";
}
function setVerdict(v) {
  const id = ITEMS[i].id; labels[id] = Object.assign(labels[id] || {}, {verdict: v}); save();
  if (v === "neither") { render(); $("correct").focus(); } else { i = Math.min(i + 1, ITEMS.length - 1); render(); }
}
["correct","note"].forEach(f => $(f).addEventListener("input", e => {
  const id = ITEMS[i].id; labels[id] = Object.assign(labels[id] || {}, {[f]: e.target.value}); save();
}));
$("prev").onclick = () => { i = Math.max(0, i - 1); render(); };
$("next").onclick = () => { i = Math.min(ITEMS.length - 1, i + 1); render(); };
$("nextU").onclick = () => { const j = ITEMS.findIndex(x => !(labels[x.id] && labels[x.id].verdict)); if (j >= 0) { i = j; render(); } };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") { if (e.key === "Enter") { i = Math.min(ITEMS.length - 1, i + 1); render(); } return; }
  if (e.key >= "1" && e.key <= "5") setVerdict(VERDICTS[+e.key - 1][0]);
  else if (e.key === "ArrowLeft") $("next").click(); else if (e.key === "ArrowRight") $("prev").click();
});
$("export").onclick = () => {
  const blob = new Blob([JSON.stringify(labels, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = EXPORT_NAME; a.click();
};
$("imp").onchange = e => { const r = new FileReader(); r.onload = () => {
  try { labels = Object.assign(labels, JSON.parse(r.result)); save(); render(); } catch (err) { alert("Not a labels file"); } };
  r.readAsText(e.target.files[0]); };
render();
</script></body></html>
"""


def write_page(path: Path, items: list[dict], storage_key: str, export_name: str,
               prev: dict[str, str] | None = None):
    page_items = []
    for d in items:
        a, b = ab_spans(d)
        page_items.append({"id": d["id"], "left": d["left"], "right": d["right"], "a": a, "b": b,
                           "crm_names": d.get("crm_names") or [], "prev": (prev or {}).get(d["id"])})
    html = (PAGE.replace("__ITEMS__", json.dumps(page_items, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__STORAGE_KEY__", storage_key).replace("__EXPORT_NAME__", export_name))
    path.write_text(html, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def refresh(out: Path, diffs: list[dict]):
    """Re-map round-1 calibration items onto the current build_diffs output.
    Unchanged items keep their id (so their hand labels still apply);
    items now filtered out (style/filler/garbled) or split into pieces drop
    out, and the pieces of a split item are added in their place."""
    round1_path = out / "calibration_items_round1.jsonl"
    if not round1_path.exists():
        shutil.copy(out / "calibration_items.jsonl", round1_path)
    round1 = [json.loads(line) for line in open(round1_path, encoding="utf-8")]
    by_id = {d["id"]: d for d in diffs}
    by_row = defaultdict(list)
    for d in diffs:
        by_row[d["row_id"]].append(d)
    kept, status = [], Counter()
    for old in round1:
        cur = by_id.get(old["id"])
        if cur is not None:
            status["filtered: " + cur["stratum"] if cur["stratum"] in NOT_JUDGED else "unchanged"] += 1
            if cur["stratum"] not in NOT_JUDGED:
                kept.append(cur)
            continue
        pieces = [d for d in by_row[old["row_id"]] if old["start"] <= d["start"] and d["end"] <= old["end"]]
        status["split"] += 1
        kept += [d for d in pieces if d["stratum"] not in NOT_JUDGED]
    write_jsonl(out / "calibration_items.jsonl", kept)
    print(f"Round-1 items: {dict(status)}; {len(kept)} judgeable items now in calibration_items.jsonl")


def review(out: Path, labels_paths: list[str], judge_dir: Path, models: list[str]):
    """A page with only the items needing another look: ones where all the
    given judges agree with each other but not with the person (the likeliest
    labelling slips), and new pieces of split items that have no label yet.
    Shows the person's earlier answer."""
    items = [json.loads(line) for line in open(out / "calibration_items.jsonl", encoding="utf-8")]
    labels = merge_labels(labels_paths)
    verdicts = {m: load_cache(cache_path(judge_dir, m)) for m in models}
    picked, prev, why = [], {}, Counter()
    for it in items:
        judged = [to_label(it, verdicts[m][it["id"]]["verdict"]) for m in models if it["id"] in verdicts[m]]
        agreed = judged[0] if len(judged) == len(models) and len(set(judged)) == 1 else None
        if it["id"] not in labels:
            picked.append(it)
            why["unlabelled"] += 1
        elif agreed and to_label(it, labels[it["id"]]["verdict"]) != agreed:
            picked.append(it)
            prev[it["id"]] = labels[it["id"]]["verdict"].replace("same", "either")
            why["disagrees with judges"] += 1
    write_page(out / "label_review.html", picked, "calibration_labels_review_v2", "calibration_labels_review.json", prev)
    print(f"{len(picked)} items to review: {dict(why)} -> {out / 'label_review.html'}")


def main():
    args = parse_args()
    out = Path(args.output_dir)
    diffs = [json.loads(line) for line in open(args.diffs, encoding="utf-8")]
    if args.refresh:
        return refresh(out, diffs)
    if args.review:
        return review(out, args.labels.split(","), Path(args.judge_dir), args.models.split(","))
    items = sample([d for d in diffs if d["stratum"] not in NOT_JUDGED], args.seed)
    write_jsonl(out / "calibration_items.jsonl", items)
    write_page(out / "label_calibration.html", items, "calibration_labels_v2", "calibration_labels.json")
    print(f"{len(items)} items: {dict(Counter(d['stratum'] for d in items))}")
    print(f"Wrote {out / 'calibration_items.jsonl'} and {out / 'label_calibration.html'}")


if __name__ == "__main__":
    main()
