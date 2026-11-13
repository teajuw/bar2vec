# Metrics

Cases are Harmonix Set songs. The namer is given each song's human-annotated section boundaries and per-bar embeddings, and returns one label per section. It never sees a label. One rep per song: the pipeline is deterministic.

**Scored labels (8):** intro, verse, prechorus, chorus, postchorus, bridge, outro, instrumental.

- Numbered variants merge: verse2 counts as verse.
- inst, instrumental, solo, break and transition count as instrumental.
- Silence and rare labels (seen fewer than 5 times on the tuning songs, such as chorusinst, breakdown, quietchorus) are named but not scored: 85 of 3,167 sections.

**macro-F1 (target):** F1 for each of the eight labels over all sections in a split, then the plain average of the eight, 0 to 100. Each label counts equally, so bridge matters as much as chorus. Computed by `scripts/nm_report.py`; it is a pooled score, so it has no per-song value.

**Share of sections right (guardrail; `acc` in the per-song rows):** scored sections named correctly, 0 to 1 per song.

**Noise:** 95% bootstrap intervals over songs (2,000 resamples, fixed seed). A change is beyond noise when the paired interval of its difference excludes zero.

**Reference points (153 train / 154 test songs):**

| Namer | macro-F1 train / test | sections right train / test |
|---|---|---|
| The annotation labels themselves (harness check) | 100 / 100 | 100 / 100 |
| Every section "chorus" | 6.5 / 6.5 | 35.5 / 35.1 |
| Uniform random label | 10.9 / 11.1 | 12.8 / 12.6 |

The per-case report (`report.html`) shows share of sections right per song, because macro-F1 is pooled.
