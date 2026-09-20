Schema of `build_dataset.py`'s output, then one row of each type.

Assembled rows are per-channel (one row per `(call_id, channel)`), not one
row per call with both channels merged -- this matches how the
`callcc-test-1k` benchmark and (per the team) actual production serving are
structured: one channel's speech is corrected at a time, not both speakers'
turns jumbled into a single request.

## Column schema

| Column | Chunked row | Assembled row |
|---|---|---|
| `call_id` | ✓ | ✓ |
| `channel` | the segment's channel | the channel this row covers |
| `assembled` | `false` | `true` |
| `start`, `end`, `duration` | segment timing | — (not present; timing lives per-turn instead) |
| `turns` | — | list of per-segment dicts (start, end, text_whisper, text_soniox, text, conf_mean, conf_min, wer_whisper, bucket) for this channel |
| `text_whisper` | segment ASR output | full-channel ASR output (joined) |
| `text_soniox` | segment punctuated gold | full-channel punctuated gold (joined) |
| `text` | segment no-punctuation gold | full-channel no-punctuation gold (joined) |
| `conf_mean`, `conf_min`, `wer_whisper` | segment values | averaged/min across this channel's usable segments |
| `bucket` | segment label | — (per-turn only) |
| `word_diff_pct`, `overlap_pct` | segment-level metric | full-channel-level metric |
| `crm_context` | always `null` | full CRM JSON, or `null` if no match/probe record |
| `seg_count_matches_meta` | — | diagnostic bool (this channel's segment count vs. meta's `n_seg_c0`/`n_seg_c1`) |

## Chunked row example

Illustrative -- `call_id` and the transcript text below are synthetic
placeholders, not real call content (`callcc-2k` is a gated dataset; real
excerpts don't belong in a public doc).

```json
{
  "call_id": "example-call-id-001",
  "channel": 0,
  "assembled": false,
  "start": 0.24, "end": 10.62, "duration": 10.38,
  "text_whisper": "سلام وقت بخیر من نمونه هستم چطور می‌تونم کمکتون کنم",
  "text_soniox": "سلام، وقت بخیر. من نمونه‌پور هستم، چطور می‌تونم کمکتون کنم؟",
  "text": "سلام وقت بخیر من نمونه‌پور هستم چطور می‌تونم کمکتون کنم",
  "conf_mean": 0.8585, "conf_min": 0.2533, "wer_whisper": 0.2143,
  "bucket": "informative",
  "word_diff_pct": 16.67, "overlap_pct": 91.67,
  "crm_context": null
}
```

## Assembled row example (truncated `turns`, full channel text)

Same caveat -- synthetic placeholder content, not a real call.

```json
{
  "call_id": "example-call-id-002",
  "channel": 0,
  "assembled": true,
  "turns": [
    {"start": 0.3, "end": 7.14,
     "text_whisper": "نمونه هستم چطور می‌تونم راهنمایی‌تون کنم ...",
     "text_soniox": "نمونه‌پور هستم، چطور می‌تونم راهنماییتون کنم؟ ...",
     "conf_mean": 0.948, "conf_min": 0.621, "wer_whisper": 0.4, "bucket": "informative"},
    { "...more turns for this channel..." }
  ],
  "text_whisper": "نمونه هستم چطور می‌تونم راهنمایی‌تون کنم ... باشه خیلی ممنون",
  "text_soniox": "نمونه‌پور هستم، چطور می‌تونم راهنماییتون کنم؟ ... خداحافظ.",
  "text": "نمونه‌پور هستم چطور می‌تونم راهنماییتون کنم ... مرسی خداحافظ",
  "conf_mean": 0.9536, "conf_min": 0.3048, "wer_whisper": 0.3201,
  "word_diff_pct": 4.47, "overlap_pct": 86.18,
  "crm_context": null,
  "seg_count_matches_meta": false
}
```

Note `crm_context` is `null` here too -- this particular call has no CRM match (as established earlier). `seg_count_matches_meta: false` reflects the known meta pre-merge segmentation mismatch (diagnostic only, doesn't affect the row).
