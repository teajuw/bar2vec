# Brief: song structure from audio-text embeddings

Written 2026-11-11 as "if this were a paper, what would it be about". Nothing in the How section has been run yet.

**What:** Cut songs into sections and name each section in plain English using only off-the-shelf audio-text embeddings, tune that pipeline against published section annotations, and compare it with methods built for the task.

## So what

### The question a paper would answer

Models such as CLAP and MuQ-MuLan were trained to match audio clips with text descriptions. Nobody told them where a verse ends. The question is how much song structure is already sitting in their embeddings, how far tuning with no training can take it, and whether the same embeddings can also say what each section is.

### What is already published, and what is left

Checked on 2026-11-11; one paper changes what the contribution can be.

- **Finding the cuts without training is done.** A March 2026 study (Marmoret, arXiv 2603.27218) tested nine frozen audio models, including CLAP and MuQ, with the same change-score method this project uses, and reported that it generally beat recent trained probes on the standard datasets. A paper claiming "embeddings can find section boundaries" would be repeating it. That study covers boundaries only.
- **What to take from that study** (local copy at `papers/marmoret-2026-unsupervised-embeddings-music-structure.pdf`, not in the repository; code at https://github.com/ax-le/msa_deep_embeddings):
  - **Embed bars, not fixed windows.** It finds downbeats with a beat tracker (Beat This!) and makes one embedding per bar, so cuts land on bar lines. This is how it competes at the half-second tolerance.
  - **A better cutting method.** Its Correlation Block-Matching method beat the change-score method (Foote) in 33 of 36 comparisons.
  - **The text-matched space is worse for cuts.** On Harmonix, MuQ's audio-only embeddings scored 52.0 / 69.7 (half-second / three-second) against 45.4 / 67.9 for the MuQ-MuLan text-matched embeddings used here. That argues for two encoders: an audio-only one to place cuts, an audio-text one to name sections.
  - **Reference scores on Harmonix:** CLAP 49.3 / 69.2; plain audio features with the Foote method 47.0 / 72.4; best model (MATPAC++) 55.3 / 73.8.
  - **Stricter scoring.** It recommends dropping each song's first and last boundary, which are free hits.
  - **A gap in its method.** Its settings were chosen on the same songs it reports, which it acknowledges. A random held-out split would be a cleaner result.
  - **It names nothing.** The study states it does not attempt to label sections.
- **Trained systems lead on accuracy.** SongFormer (2025) and All-In-One are trained on section labels and set the scores to beat. An untrained method should not be expected to match them.
- **What looks open:**
  1. **Naming sections in free text.** Trained systems choose from a fixed list (verse, chorus, bridge). Audio-text embeddings can describe a section in any words ("stripped back, vocals only"). I found no published test of this; the search was three queries, not a literature review.
  2. **Tuning without labels.** In this project the two models put their cuts in the same place 79% of the time against 48% by chance. If settings that raise that agreement also raise accuracy against human annotations, a segmenter could be tuned for any music with no labelled data.
  3. **A cleaner measurement than the published one.** The March study chose its settings on the same songs it reports. Tuning on one random set of songs and reporting on another, never opened during tuning, would give a boundary score that can be trusted to generalise.
  4. **One pipeline for search, structure and description.** The same window embeddings already serve moment search here, so structure comes at no extra embedding cost.

The honest size of this: the first open item is the paper, the second is a strong supporting result, and the boundary scores are a required table, not a contribution.

### The same idea in language and vision

| Field | The equivalent | How close |
|---|---|---|
| Language | Topic segmentation. TextTiling (1997) cuts a document where neighbouring blocks of text stop resembling each other; "semantic chunking" in today's retrieval systems does the same with embeddings | The same algorithm on a different kind of token |
| Vision | Shot and scene detection in video from per-frame embeddings; object discovery from the similarity between image patches in self-supervised models, with no labels | Same principle: structure appears in features trained for something else |
| Vision | Open-vocabulary segmentation: find regions first, then name each with a text-matching model | The direct analogue of naming sections in free text |

## How

### Ground truth

- **Use published annotations, not Genius.** Genius has section names in order (verse, chorus) but no timestamps, so it cannot score where a cut falls. It can serve as a weak check on the order and count of sections.
- **Harmonix Set:** 912 pop songs with section times. Audio is not included; it lists a YouTube link per song and a score for how well that video matches the original.
- **SALAMI:** over 1,300 tracks across genres, annotations free to use, audio not included.
- **SongFormBench:** 300 expert-checked songs, 200 of them from Harmonix. Whether audio is included is unchecked.
- **A small set of your own:** 10 to 20 songs from the playlist, hand-marked. Needed only for the free-text naming test, which has no existing ground truth.

No new large dataset is needed. The work is fetching audio and discarding songs whose audio does not match the annotated version.

### Score

The field's standard measures: the share of cuts within 0.5 seconds and within 3 seconds of a human-marked cut, and how well the grouping of sections matches. The current five-second spacing cannot compete at 0.5 seconds, so finer spacing is the first change.

### Levers, cheapest first

| Stage | Levers | Cost per try |
|---|---|---|
| 1. Settings, no training | Window length; spacing between windows; aligning windows to bars; which model, or both combined; how far the change score looks each side; cut threshold; grouping threshold | Seconds to minutes once embeddings are cached |
| 2. Small trained head on frozen embeddings | A classifier or small transformer over the window vectors, trained on section labels | Minutes |
| 3. Fine-tune the embedding model (LoRA) | Teach the model that windows in the same section belong together | Hours, plus a risk of harming text matching |

Stage 3 is last on purpose. It is the most expensive, it gives up the "no training" claim, and it could weaken the text side that the naming contribution depends on. It is worth trying only if stages 1 and 2 plateau.

### Is it a hill climb or a random search?

Stage 1 is a hill climb. There is a repeatable score, each try is cheap, and the levers are few and interpretable. The discipline it needs:

- Split songs into a tuning set and a held-out set chosen at random, and report only the held-out score.
- Change one thing per round and try several values of it.
- Keep the untuned pipeline as the control in every comparison.

Stages 2 and 3 are ordinary model training, where a held-out set matters even more.

### Comparisons

- A control using plain audio features instead of a pretrained model, to show what the embeddings add.
- The classical untrained methods (Foote, spectral clustering), available in the MSAF package.
- Audio-only pretrained models (MERT, MusicFM) through the same pipeline.
- Trained systems (All-In-One, SongFormer) as the ceiling.

### Starting point, measured 2026-11-11

The breakdown pipeline exactly as built (ten-second windows every five seconds, hand-set thresholds), scored on 39 Harmonix songs. F is the standard boundary score out of 100; "trimmed" drops each song's first and last boundary.

| Method | F within 3 s | F within 3 s, trimmed | F within 0.5 s |
|---|---|---|---|
| Control: same number of cuts, evenly spaced | 37.8 | 22.8 | 20.1 |
| CLAP, as built | 50.8 | 38.8 | 23.9 |
| MuQ-MuLan, as built | 49.0 | 36.4 | 23.5 |
| CLAP, cuts moved 2.5 s later | 55.0 | 44.1 | 23.3 |
| MuQ-MuLan, cuts moved 2.5 s later | 55.0 | 43.8 | 23.8 |
| Published: CLAP per bar with a tuned method (Marmoret) | 69.2 | not reported for CLAP | 49.3 |

- **The cuts carry real signal:** 13 to 21 points above evenly spaced cuts at the three-second tolerance.
- **They are well short of the published result:** about 14 points at three seconds and about 25 at half a second. The half-second score is barely above the control, as expected from a five-second grid.
- **One correction is already visible.** A cut was being reported at the start of the window after the change; the change actually sits in the middle of the overlap, 2.5 seconds later. That shift is derived from the window geometry, not fitted, and is worth 4 to 6 points.
- **The sample is small and filtered.** 200 songs were drawn at random; 39 were kept because their YouTube audio matched the annotated length within two seconds. Most links are a different edit of the song. The kept songs may still start slightly early or late, which would lower every score, the half-second one most.
- **Not a like-for-like comparison.** The published figure is on all 912 songs with the original audio.

### What could sink it

- **Free-text section names may be too unreliable.** In this project the cuts looked steadier than the words, and the vocals descriptions were often wrong. If naming fails, the main contribution goes with it.
- **Audio matching.** Annotations are tied to a specific recording; a different edit shifts every timestamp.
- **Agreement may not track accuracy.** The label-free tuning idea is a hypothesis.
- **Ten-second windows may be too coarse** for cuts that need to land within half a second, even with fine spacing.

## Sources

- Marmoret, "Unsupervised Evaluation of Deep Audio Embeddings for Music Structure Analysis": https://arxiv.org/abs/2603.27218
- SongFormer: https://arxiv.org/abs/2510.02797 and https://github.com/aslp-lab/songformer
- SongFormBench: https://huggingface.co/datasets/ASLP-lab/SongFormBench
- EDMFormer (trained, electronic music): https://arxiv.org/abs/2603.08759
- Harmonix Set: https://github.com/urinieto/harmonixset
- SALAMI loader notes: https://mirdata.readthedocs.io/en/latest/_modules/mirdata/datasets/salami.html
