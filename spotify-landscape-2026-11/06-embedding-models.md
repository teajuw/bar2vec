# Is there something better than CLAP? (2026-11-11)

Quick pass: about seven searches, reading summaries rather than full papers. Numbers come from papers and secondary summaries, several of them self-reported by a competing model's authors. Treat them as directional.

## Short answer

Yes, probably, for music specifically. **MuQ-MuLan** (Tencent, 2025) is the best-supported alternative for matching text to music. But the margin depends on the benchmark, the released weights are weaker than the paper's, and `vibe` already uses CLAP's music-tuned checkpoint rather than the default one most papers compare against. The honest move is to run both on the same audio and queries.

## Candidates for text-to-audio search

| Model | What it is | Evidence it beats CLAP | Catches |
|---|---|---|---|
| **LAION-CLAP** (current) | General audio-text model; `vibe` uses the `music_audioset` checkpoint | Baseline | Papers usually test the default checkpoint, not the music one |
| **MuQ-MuLan** | Music-specific audio-text model | Zero-shot tagging on MagnaTagATune: 79.3 vs 73.9 ROC-AUC. GTZAN genre on the MAEB benchmark: 85.4 vs 61.3. Text-to-music R@10 on WikiMT-X: 12.1 vs 6.9 (reported by a third paper) | Weights are CC-BY-NC (non-commercial). Needs 24 kHz input. Maintainers say the released model, trained on Million Song Dataset, may not match the paper |
| **CLaMP 3** | Music retrieval across audio, sheet music and text; ACL 2025 | Reported to beat CLAP on English text-to-music retrieval (MRR 0.199 vs 0.131 on one set, from a secondary review) | Weights released; licence not confirmed in this pass. MuQ-MuLan comparison not found |
| **HeartCLAP** | Part of HeartMuLa, January 2026 | Claims to beat both | Self-reported by its authors; not looked at further |

Where CLAP holds its own:

- **Perceptual similarity:** on an instrument-similarity listening test, CLAP and MuQ-MuLan are nearly tied (71.9% vs 72.4% agreement with human listeners).
- **General audio:** CLAP is far ahead on non-music sound (ESC-50: 86.9 vs 62.6). Irrelevant for a music library.

## Audio-only models (no text side)

MERT, MusicFM, MuQ and OMAR-RQ are self-supervised music encoders. They cannot take a text query, so they don't replace CLAP for "describe a vibe." They are stronger on low-level properties such as key and pitch, and they are what structure-analysis models are built on. Useful later for song-to-song similarity and for feature extraction.

## Finding section boundaries

**SongFormer** (open source, built on MuQ and MusicFM) labels intro, verse, pre-chorus, chorus, bridge, instrumental and outro with timestamps. That is the "section finder" tool from the set-building design.

Caveat: its labels are pop-oriented. A follow-up preprint reports it does poorly on EDM (accuracy 0.15 on an EDM set versus 0.88 for an EDM-specific model). If the library leans electronic, fixed windows may beat section labels.

## Unconfirmed: how vibe's CLAP call treats a full song

`vibe` passes whole files to CLAP with fusion off. Downstream projects suggest that mode truncates to a random ten-second crop, but I did not find LAION's own documentation confirming it. Check `hook.py` in the LAION-AI/CLAP repo before relying on it.

## Recommendation

1. Embed the same ten-second windows with **both** CLAP (music checkpoint) and MuQ-MuLan.
2. Score both on the same hand-written test queries.
3. Keep whichever wins on this library. The non-commercial licence on MuQ-MuLan is fine for a personal project and a portfolio demo.
4. Leave SongFormer for a second pass, after fixed windows are working.

## Sources

- https://arxiv.org/html/2501.01108v1 (MuQ and MuQ-MuLan paper)
- https://www.github.com/tencent-ailab/MuQ (code, licence, input requirements)
- https://arxiv.org/pdf/2602.16008 (MAEB: Massive Audio Embedding Benchmark)
- https://arxiv.org/pdf/2601.10547 (HeartMuLa, source of the WikiMT-X comparison)
- https://www.emergentmind.com/topics/pretrained-text-audio-embeddings-clap-and-muq-mulan (secondary summary)
- https://arxiv.org/html/2502.10362v3 (CLaMP 3)
- https://arxiv.org/pdf/2306.10548 (MARBLE benchmark)
- https://arxiv.org/html/2509.06936 (autotagging benchmark across seven encoders)
- https://arxiv.org/html/2510.02797v2 and https://github.com/aslp-lab/songformer (SongFormer)
- https://arxiv.org/html/2603.08759v1 (EDMFormer)
- https://github.com/LAION-AI/CLAP
