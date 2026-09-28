"""Inference on whole transcripts with a model trained on short windows.

A transcript is cut into overlapping windows (WINDOW words, starting every
STRIDE words). Each word position is *owned* by exactly one window -- the one
where it sits in the middle part (the "core"), so every word is decided with
at least (WINDOW - STRIDE) / 2 words of that window on each side, except at
the transcript's own start and end. The model runs on every window; an edit
is kept only from the window that owns its first word, and only if it lies
fully inside that window. Kept edits are shifted to transcript positions,
validated together (overlaps: first one wins), and applied to the Whisper
text -- which never changes anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from edit_format import parse_edits, render_input
from edits import Edit, apply_edits, validate_edits

WINDOW = 50
STRIDE = 25


@dataclass
class Window:
    start: int
    end: int
    core_start: int
    core_end: int


def plan_windows(n_words: int, window: int = WINDOW, stride: int = STRIDE) -> list[Window]:
    """Overlapping windows over n_words whose cores partition [0, n_words)."""
    if n_words <= window:
        return [Window(0, n_words, 0, n_words)]
    margin = (window - stride) // 2
    starts = list(range(0, n_words - window + 1, stride))
    if starts[-1] + window < n_words:
        starts.append(n_words - window)  # last window flush with the end
    wins = []
    for k, s in enumerate(starts):
        core_start = 0 if k == 0 else wins[-1].core_end
        core_end = n_words if k == len(starts) - 1 else min(s + margin + stride, n_words)
        wins.append(Window(s, s + window, core_start, max(core_start, core_end)))
    return wins


@dataclass
class TranscriptResult:
    corrected: str
    edits: list[Edit]
    windows: int
    rejected_lines: list[tuple[str, str]] = field(default_factory=list)
    dropped_outside_core: int = 0


def build_window_prompts(transcripts: list[str], crm_names: list[list[str]], window: int = WINDOW,
                         stride: int = STRIDE) -> tuple[list[tuple[int, Window]], list[str]]:
    """(transcript index, window) per model call, and the user text for each."""
    index, users = [], []
    for t, (text, names) in enumerate(zip(transcripts, crm_names)):
        words = text.split()
        for w in plan_windows(len(words), window, stride):
            index.append((t, w))
            users.append(render_input(words[w.start:w.end], names))
    return index, users


def merge_outputs(transcripts: list[str], index: list[tuple[int, Window]], outputs: list[str]) -> list[TranscriptResult]:
    """Model outputs per window -> one corrected transcript per input."""
    per_t: dict[int, list] = {t: [] for t in range(len(transcripts))}
    for (t, w), out in zip(index, outputs):
        per_t[t].append((w, out))
    results = []
    for t, text in enumerate(transcripts):
        words = text.split()
        kept, rejected, outside = [], [], 0
        for w, out in per_t[t]:
            parsed = parse_edits(out, words[w.start:w.end])
            rejected += parsed.rejected
            for e in parsed.edits:
                g = Edit(e.start + w.start, e.end + w.start, e.original, e.replacement)
                if w.core_start <= g.start < w.core_end:
                    kept.append(g)
                else:
                    outside += 1  # another window owns this position
        valid, invalid = validate_edits(words, sorted(kept, key=lambda e: e.start))
        rejected += [(" ".join(e.replacement), reason) for e, reason in invalid]
        results.append(TranscriptResult(apply_edits(text, valid), valid, len(per_t[t]), rejected, outside))
    return results
