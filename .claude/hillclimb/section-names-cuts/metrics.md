# Metrics

- **macro_f1_time** (target): the cutter cuts the song, the classifier names the cutter's sections, and every second of an annotated, scored section is compared with the name given to that second. F1 per label from seconds pooled over songs, averaged over the eight labels, x100. ±95% bootstrap over songs. It is a pooled number, so it lives in each variant's `summary.json`, recomputed from `conf_<split>.npy` (songs x annotated label x given name, in seconds).
- **seconds_right** (guardrail): the share of scored seconds given the right name. Per song in `results.jsonl`, pooled in `summary.json`.

| Variant | Tuning, out-of-fold | Held-out | Held-out vs baseline |
|---|---|---|---|
| baseline: fitted on annotated sections | 50.1 ±3.0 / 68.4 | 53.1 ±2.9 / 67.9 | |
| v1: fitted on annotated sections plus the cutter's sections | 52.8 ±3.1 / 70.4 | 53.2 ±3.1 / 68.4 | +0.1 ±2.2 / +0.5 ±1.5 |
| v2: section classifier with per-bar votes as inputs | 53.8 ±3.4 / 70.5 | 55.8 ±3.2 / 69.7 | +2.7 ±2.3 / +1.8 ±1.8 |
