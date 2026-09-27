"""Corpus word frequencies, shared by the edit-method scripts (the rules'
and crm_snap's rarity gates, and fix_rate_recoverable)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared modules

from evaluate import _load_hub_columns_pruned  # noqa: E402

CACHE_DIR = Path("outputs/edit/cache")


def load_corpus_freq(train_repo: str, train_fraction: float | None = None, seed: int = 42,
                     cache_dir: Path = CACHE_DIR) -> Counter:
    """Word frequencies over a training set's `text` targets -- the same
    Counter train.py builds, including its train_fraction subsample
    (data.load_sft_dataset's shuffle(seed).select(n), which depends only on
    the seed and row count, so a text-only load gives the same subset)."""
    cache = cache_dir / f"corpus_freq_{train_fraction or 1.0}_{seed}.json"
    if cache.exists():
        return Counter(json.loads(cache.read_text(encoding="utf-8")))
    print(f"Counting word frequencies over {train_repo} (fraction={train_fraction}, cached to {cache})...")
    ds = _load_hub_columns_pruned(train_repo, ["text"], "train")
    if train_fraction is not None and train_fraction < 1.0:
        ds = ds.shuffle(seed=seed).select(range(round(len(ds) * train_fraction)))
    freq = Counter()
    for text in ds["text"]:
        freq.update(text.split())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(freq, ensure_ascii=False), encoding="utf-8")
    return freq
