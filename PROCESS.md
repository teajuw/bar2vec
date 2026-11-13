# bar2vec process log: from "music as an AI domain" to a tested section cutter

The project was named `moment-search` until 2026-11-13, when it was renamed `bar2vec`: one vector per bar is the core step, and bar alignment was the largest gain.

Written 2026-11-13 as source material for writeups. It records what was done, in order, why each step was taken, what was decided by whom, and what is still unverified. Numbers are copied from the files named beside them; where this log and those files disagree, the files win.

All work happened in one working session, 2026-11-11 to 2026-11-13, on a 64 GB Mac mini.

## One-paragraph version

The starting idea was to treat music as an ordinary domain for standard AI techniques (embeddings, chunking, search). A moment-search prototype on one playlist led to an automatic song breakdown, which raised the question of how good the section cuts were. That became a hill climb against published section annotations (the Harmonix Set), with a random train / held-out split and a locked final test. The cutter went from 34.5 to 72.2 on held-out songs with no training, and the final locked test confirmed it (35.2 to 72.7 on 180 songs never touched during the climb). A small supervised add-on reaches 75.4.

## Where the detailed records are

| What | Where |
|---|---|
| Round-by-round table and reasoning for the climb | `.claude/hillclimb/song-cuts/narrative.md` |
| Climb state: split ids, notes, final result | `.claude/hillclimb/song-cuts/_state.json` |
| Per-song scores and run logs for every variant | `.claude/hillclimb/song-cuts/<variant>/` |
| Final locked test runs | `.claude/hillclimb/song-cuts-final/` |
| First climb attempt on 97 songs (archived) | `.claude/hillclimb/song-cuts-97/` |
| Spectrogram comparison row | `.claude/hillclimb/song-cuts-mel/` |
| Project brief and paper-style brief | `brief.md`, `brief-song-structure.md` |
| Prototype results and file layout | `README.md` |
| Spotify feature and competitor research, idea list | `spotify-landscape-2026-11/` |
| Benchmark paper (Marmoret et al. 2026) | https://arxiv.org/abs/2603.27218 (PDF kept locally, not in the repository) |
| Music language model notes and ten papers | `papers/music-language-models/` |
| The cutter and its settings | `scripts/cuts.py`, `cuts.json` |

## Timeline

### 1. Framing (2026-11-11, evening)

- Started from existing GitHub projects on Spotify and music embeddings, and the question of whether they were one idea or several.
- Did a research pass on what Spotify and competitors already ship (`spotify-landscape-2026-11/`).
- Reframed: music as an untapped domain for normal AI techniques, with embeddings as the main tool. Produced a list of ideas, each tied to the technique it borrows from (`05-ai-technique-ideas.md`). Two stood out: set building and moment search.

### 2. Moment-search prototype

- Built on one 82-track playlist. Every song is cut into 10 s windows every 5 s; each window gets an audio-text embedding (CLAP and MuQ-MuLan); a text query returns windows with timestamps.
- Scored with 18 draft queries. Finding: using all windows beats one random crop per song; for whole-song queries the mean of windows is as good as the best window; the one moment-specific query favoured moment search. Details in `README.md`.
- Caveat that still stands: the queries were written without listening, and nobody has done a listening pass on the result pages.

### 3. Next steps from embeddings

- Wrote `brief.md` after an interview about what the project is for.
- Built an embedding-to-English readout (`readout.html`) and a song breakdown (`breakdown.html`) that cuts each song where neighbouring windows change most. The breakdown used no outside knowledge of sections.
- The breakdown prompted the idea that drove the rest of the work: published section annotations exist, so the cutter can be tuned against ground truth.

### 4. Setting up the climb (2026-11-12, early)

- Dataset: the Harmonix Set (912 songs with section boundaries and beat annotations). Audio is not distributed, so songs were matched to downloads.
- First attempt used 97 songs whose audio length matched the annotation. One round was inconclusive: the noise on 48 held-out songs was about ±4.7, larger than any quick fix. Conclusion: the limit was the number of songs.
- Recovered more songs by aligning downloaded audio to the annotated recordings using the spectrograms Harmonix ships. This gave 307 aligned songs, split at random and balanced by genre into 153 train and 154 held-out.
- A further 180 songs aligned after the split was drawn were locked as a final test set, never embedded or scored until the end.
- Target: boundary F within 3 s with the first and last boundary dropped (they are free hits). Guardrail: the same score at half a second. Scored with `mir_eval`.
- Rules, from the `hillclimb` skill: analysis reads train songs only; one hypothesis per round; keep a change only if train and held-out both rise beyond noise; state saved on disk each round.
- Agreed loop structure: an inner loop of quick fixes that need no re-embedding, run unattended until saturated; an outer loop of expensive changes (re-embedding or a new method), with a check-in before each. After each kept outer step, run the inner loop again. Stop when two outer steps in a row fail to beat noise.

### 5. The "so what" check and the benchmark paper

- Midway, the question was raised of why not simply use an existing method. A March 2026 paper (Marmoret et al.) had already shown frozen audio embeddings can find section boundaries without training. This changed what the work could claim.
- What was taken from the paper: embed bars instead of fixed windows; its block-matching cutting method; the finding that audio-only encoders beat text-matched ones for cuts; its stricter scoring.
- What the paper left open: its settings were chosen on the same songs it reports. A random held-out split and a locked final test are this project's methodological addition.
- A plain-spectrogram comparison row was added so the embedding result had a non-learned reference.

### 6. The climb, round by round

Held-out 3-second score on 154 songs. Full table with intervals in `narrative.md`.

| Round | Change | Held-out | Decision |
|---|---|---|---|
| 0 | Baseline: CLAP, 10 s windows every 5 s | 34.5 | |
| 1 | Report each cut 2.5 s later (window geometry) | 42.6 | kept |
| 2 | Lower the cut threshold | 43.4 | reverted, within noise |
| 3 | Place cuts between grid points | 50.8 | kept |
| 4 | Window every 2 s (outer) | 50.6 | reverted |
| 5 | One embedding per bar, human-annotated bars | 67.7 | upper-bound check, not adoptable |
| 6 | Bars from a tracker trained on these songs | 68.2 | check, not adoptable |
| 7 | Bars from tracker models that never saw the song (outer) | 67.9 | kept |
| 8 | Block matching alone (outer) | 67.8 | reverted |
| 9 | Block matching proposes, change score vetoes | 70.0 | kept by user decision; gain +2.1 ±2.2 was just inside noise |
| 10 | MATPAC++ encoder instead of CLAP (outer) | 72.2 | kept |
| 11 | Three bars of context per embedding (outer) | 68.9 | reverted |
| 12 | Learned keep/drop model on the proposals (outer) | 75.4 | passes; supervised, reported separately |

Points worth keeping for a writeup:

- **Bar alignment was the largest single gain** (+17 on the target; half-second 9.8 to 43.7). Rounds 5 to 7 were designed to separate the gain from any leak: human bars, a tracker that had trained on these songs, and a fair tracker all scored the same on the target.
- **The fair tracker matters.** Harmonix is in Beat This!'s training data. Each song was tracked by the cross-validation model that did not train on it.
- **Block matching alone did not beat the change score here**, unlike in the paper. The combination did: block matching proposes cuts and the change score vetoes weak ones.
- **Round 9 was a judgment call.** By the keep rule it was unproven on the target. It was kept for its half-second gain (+6.1 ±2.6). Rounds 9 and 10 together clear noise against round 7 (+4.3 ±2.7).
- **The encoder swap is confounded.** CLAP needed each bar repeated to fill 10 s; MATPAC++ takes the bar alone. Round 10 changed both at once.
- **Inner loops saturated twice**, on the fixed-window setup after round 3 and on the final setup after round 10. Sweeps of every quick setting stayed within about a point.
- **Negative results on train only, no round spent:** adding a chroma change score; a 5 s guard between compared sides; other encoder layers (the last layer was best); per-song PCA.

### 7. The last outer steps and the stop

- **Learned merge model (round 12).** Built by a subagent that saw train songs only: boosted trees over about 100 features of each proposed cut, replacing the veto. Held-out +3.2 ±1.1. It is trained on Harmonix annotations, so the cutter is no longer training-free with it.
- **Beat-level placement.** Diagnosis on train: 98% of annotated boundaries sit on a tracked beat but only 78% on a tracked bar start; 15 of 153 songs have the bar grid on the wrong beat. Upper bound from fixing placement: about +3 on the target and +13 at half a second. Every audio-only rule tried (about 100 settings) and a learned placer failed to beat noise; a control trained on shuffled targets got about half the learned placer's gain.
- **Stop.** Round 11 and the beat-level work were two failed outer steps in a row, which met the agreed stopping rule.

### 8. Side thread: music language models

- A question about whether "tokenize, self-attend, predict the next token" carries over to sound led to a short reading pass (`papers/music-language-models/`). Only abstracts and one method section were read.
- The finding relevant here: YuE (2025), an open long-form song generator, depends on an automatic section cutter to keep structure over a full song, and needs a label per section. That gives the cutter a use beyond a playlist page and raises the priority of section naming.

### 9. Final test (2026-11-13)

The 180 locked songs were bar-tracked, embedded and scored once per row. No setting changed afterwards.

| Row | 3-second | Half-second | Same row on the 154 held-out songs |
|---|---|---|---|
| Baseline, as built | 35.2 ±2.2 | 7.0 ±1.4 | 34.5 / 6.7 |
| Training-free cutter (v10) | 72.7 ±2.6 | 53.9 ±3.4 | 72.2 / 52.5 |
| Learned merge model (v12) | 75.4 ±2.4 | 55.3 ±3.3 | 75.4 / 55.3 |

- Paired per song: v10 minus baseline +37.5 ±3.1; v12 minus v10 +2.7 ±1.3.
- The held-out scores from the climb were not inflated by twelve rounds of selection.
- One correction: the first baseline run on the final set used a newer default (cut refinement on) and gave 35.7. With the as-built setting the 307 recorded baseline grades reproduce exactly; that run is the reported row. The baseline was therefore scored twice on the final set. The mistaken run is kept in `song-cuts-final/baseline_refine_default_mistake/`.
- Before scoring, the edited runner reproduced v10 and v12 on all 307 climb songs.
- For reference, the paper reports 55.3 / 73.8 (half-second / 3-second) for MATPAC++ on all 912 Harmonix songs with its own bar tracking and with settings chosen on the reported songs. The numbers are not directly comparable: different song subsets and audio.

### 10. Section naming (2026-11-13)

A second climb, on naming sections once they are cut. Full record in `.claude/hillclimb/section-names/narrative.md`.

- Setup agreed with the user: the namer is given the annotated section boundaries (naming measured on its own); eight labels (intro, verse, prechorus, chorus, postchorus, bridge, outro, instrumental); target macro-F1 so each label counts equally; training-free first, a trained classifier as a separate row. Same train / held-out split and the same locked 180 songs.
- Baseline, matching each section to text prompts such as "the chorus of a song" with CLAP, scored 9.0, which is chance (random labels score 11). Section roles are not a sound.
- Five training-free rounds, all kept, took held-out macro-F1 to 56.0: first section is intro and last is outro (23.3); the most repeated group of sections is the chorus (34.6); a one-off section in the second half is the bridge (42.4); short repeating sections beside the chorus are pre- or post-chorus (53.0); a verse or bridge with far less singing is instrumental (56.0). Sections are grouped by MATPAC++ similarity; the only use of text is CLAP telling sung from not sung within a song.
- An analyzer on the tuning songs found the rules saturated; the largest loss is picking the wrong group as chorus (32 of 153 songs). Adding loudness to that pick gained on tuning songs and failed on held-out (-0.3 ±1.8): reverted.
- A trained classifier over about 130 per-section features, one untuned configuration, reached 66.7 on held-out (+10.7 ±4.4). Its inner loop then saturated: about 15 variants, none beyond noise.

Final test, 180 songs, scored once per row:

| Row | Macro-F1 | Sections right | Held-out during the climb |
|---|---|---|---|
| Baseline: nearest text prompt | 9.3 ±2.0 | 14.0 ±1.7 | 9.0 / 14.3 |
| Training-free rules (v5) | 55.9 ±3.9 | 63.8 ±3.7 | 56.0 / 63.3 |
| Trained classifier (v7), fitted on 150 annotated songs | 65.9 ±3.7 | 74.7 ±3.1 | 66.7 / 76.0 |

Qualifications: names were scored on human-annotated boundaries, not on bar2vec's own cuts, so end-to-end quality is unmeasured. Postchorus and the mixed instrumental label are weak (final F1 19 and 27 for the rules) and partly reflect annotator choice. The rules encode Western pop song form and are untested elsewhere.

### 10b. After the naming final test: a bar-level transformer and more labels (2026-11-14)

Prompted by the user's questions about attention and where to find more labels. The final 180 songs had already been scored for naming, so only cross-validation on the tuning songs and the 154 held-out songs were used.

- **Bar-level transformer** ("spotiformer", the user's name for it): each bar is a token and every bar attends to every other bar in the song. Cross-validated on 150 tuning songs it reached macro-F1 44.5 to 45.3, and 53.4 when also given the rule-based names, against 56.6 for the rules and 62.6 for the boosted trees. Attention does help (sections right 65.5 against 57.2 with attention removed), and the common labels match the rules, but the rare labels are not learned from 150 songs.
- **Where more labels could come from** was surveyed (the rest of Harmonix, the HookTheory set, SongFormDB's machine labels, SALAMI-pop, RWC-Pop, Isophonics, lyric-site section tags, labels from an existing segmenter). The cheapest was tried: 213 downloaded Harmonix songs that failed the alignment bar for scoring but have at least 60 s of aligned audio, used for training only with labels kept in aligned audio. That doubled the labelled data.
- **Result:** in cross-validation the extra songs helped everything (transformer +5.4 ±2.9, boosted trees +3.9 ±2.9). On the held-out songs the boosted trees with the extra songs scored 67.0 against 66.7 without, +0.3 ±2.4: no gain. The transformer stayed 10.7 behind the trees and was not scored on held-out.
- **Reading:** doubling same-dataset labels did not move held-out naming, so label count alone is not what limits the trained row at this size. Two failed outer steps in a row closed the naming climb.

### 10c. Neighbour similarity as a naming signal (2026-11-13)

From the user's listening on the playlist page: neighbouring sections with the same name sounded different, and the fault seemed to be the name, not the cut. The idea was to use block-to-block similarity as a stand-in for attention and park the learned version.

- **The flag is real.** On the playlist, 123 neighbouring pairs share a name (89 verse-verse) and only 34 of them are in the same similarity group. On the 153 tuning songs, when two neighbours in different groups are both called "verse" (127 pairs), the annotators give them the same label 9% of the time.
- **The fix was not found.** Most of these come from picking the wrong group as the chorus. Three rule families that use neighbour similarity or neighbour order were swept on the tuning songs; the best gained +0.5 ±0.6. No held-out round was spent and the rules are unchanged.
- **What it leaves:** the disagreement can mark a name as uncertain on the page even though the rules cannot say what the section is instead. The trained namer already uses its neighbours' names.

### 10d. Names end to end, on bar2vec's own cuts (2026-11-13)

A measurement, not a round: nothing was tuned, and the 180 final songs were not touched. `scripts/nm_e2e.py` runs the cutter (v10), names its sections, and scores by time: each second of an annotated section against the name bar2vec gives that second. Macro-F1 over the eight labels, 154 held-out songs (results in `.claude/hillclimb/section-names-e2e/`).

| Row | Macro-F1 by time | Seconds named right |
|---|---|---|
| Best possible names on bar2vec's cuts | 89.5 ±1.6 | 91.0% |
| Rules on annotated edges | 54.9 ±4.3 | 65.7% |
| Rules on bar2vec's cuts | 44.2 ±3.4 | 54.9% |
| Trained namer on annotated edges | 66.4 ±4.2 | 79.0% |
| Trained namer on bar2vec's cuts | 53.1 ±2.9 | 67.9% |

- **The cuts cost the rules 10.7 ±3.7 points.** 4.6 ±1.3 is misplaced edges alone (the rules' names from annotated edges carried onto the cut sections). The other 6.1 ±3.3 is the rules naming differently when given the cuts.
- **The namer is still the larger loss.** Perfect names on the same cuts would score 89.5; the rules are 45 points below that.
- **The trained namer keeps its lead on the cuts:** +8.9 ±2.9 over the rules. It was fitted on annotated edges and loses 13.3 on cuts.
- The tuning songs give the same picture for the rules (55.1 on annotated edges, 44.7 on cuts).

Acted on the same day: the playlist page now uses the trained namer (`scripts/playlist_bars_extra.py` adds per-bar CLAP and loudness for the playlist; `scripts/breakdown_bars.py --namer rules` gives the old names). 250 of 760 section names changed, and unlike neighbours both called "verse" fell from 77 to 51. A name is marked "?" when the classifier gives it less than 0.8 probability (252 of 760). On held-out songs with annotated edges, names below 0.8 are right 46% of the time and the rest 87%; the 0.8 line was picked after seeing those held-out bins, so treat the two rates as a description, not a test. The new names have not been listened to.

### 11. Duplicate songs in Harmonix (found 2026-11-13)

Harmonix contains repeats and remixes of the same song. Among the 487 songs used there are 17 title-and-artist groups (36 songs); 9 groups span splits, two pairs across train and held-out, and seven final songs have a sibling among the climb songs. This applies to the cutter results as well. For rule-based methods it can leak little, since nothing is memorised; it matters more for the trained rows (the cutter's merge model was fitted without excluding these; the trained namer excluded the three affected train songs).

## Decisions and who made them

| Decision | Made by | Reason |
|---|---|---|
| Pursue moment search first, park other ideas | user | most concrete to prototype |
| Tune the cutter against published annotations | user's idea | ground truth exists |
| Recover songs by alignment instead of accepting 97 | user | sample size was the binding limit |
| Inner / outer loop structure and stopping rule | agreed | keeps expensive steps deliberate |
| Do bar alignment before waiting for more songs | user | largest expected gain |
| Keep round 9 despite the keep rule | user | half-second gain was clear |
| Lock the 180 late-aligned songs as a final test | assistant, at the split | they arrived after the split was drawn |
| Report the merge model as a separate labelled row | assistant recommendation | it is supervised on the benchmark's annotations |
| Use the training-free cutter going forward | assistant recommendation, 2026-11-13 | see below |

Recommendation recorded 2026-11-13: use the training-free cutter (v10) as the project's cutter. The merge model's +2.7 was learned from one pop-heavy dataset, it adds nothing on hip-hop in the final test (71.6 for both, 31 songs), and the training-free claim is the part that transfers to music outside Harmonix.

## What a writeup can and cannot claim

Can claim:

- A training-free cutter built from frozen audio embeddings scores 72.7 ±2.6 (3 s) and 53.9 ±3.4 (0.5 s) on 180 Harmonix songs that played no part in tuning.
- Settings were chosen on train songs, checked on a random held-out set, and confirmed once on a locked set.
- Most of the gain came from where cuts are allowed to fall (bar starts), not from a better encoder.

Cannot claim, or must qualify:

- Anything outside Harmonix, which is Western pop, hip-hop, dance and country. Not measured.
- That this beats the paper. Song subsets and audio differ.
- That the output sounds right. No one has listened to the results pages.
- The final set is now used and cannot serve as a clean test for further tuning.
- 487 of 912 Harmonix songs were recovered; the ones that aligned may differ from the ones that did not.

## Known limits of the measurement

- The typical beat is 0.50 s, so a one-beat error sits on the half-second tolerance and counts or not depending on tempo. Similarly 3.0 s is a bar and a half at 120 BPM.
- Five train songs show 70 to 167 ms residual misalignment, which caps their half-second scores.
- About 210 of 390 false cuts on train are 4 or 8 bars from a true boundary: the annotators merged a repeated block that the cutter splits. Some of these are defensible cuts.
- Tracker agreement with annotated downbeats on the final 180 songs: mean F 0.859.

## Open items

- Done 2026-11-13: the bar2vec cutter is on the playlist `breakdown.html` as a third row under the two original cutters (`scripts/playlist_bars.py`, `scripts/breakdown_bars.py`). Bars come from the tracker's default model. Not yet listened to. The letter grouping threshold (0.35) is untuned, and 15 of 81 songs have a median bar under 1.5 s or over 3.3 s, which suggests the tracker picked double or half tempo on some.
- Listening pass on `similar.html`, `readout.html` and `breakdown.html`.
- Done 2026-11-13 for fixed labels (section 10); free-text naming and naming on bar2vec's own cuts are still open.
- Done 2026-11-13: the playlist `breakdown.html` shows a name on each bar2vec section, from the training-free rules (sung-or-not taken from the CLAP windows, since the playlist has no per-bar CLAP embeddings). Not listened to. These names sit on bar2vec's own cuts, a combination that has not been scored anywhere.
- Section-based song similarity (matching sections instead of averaging a whole song).
- Separate the encoder gain in round 10 from removing the repeat-to-10-s step.
- Test on music outside Harmonix.
- `data/queries.json` has never been checked by ear.

## Reproducing the final rows

Run from the project root. Large files (`audio_hx/`, `data/*.npz`, `external/`, `models/`) are not in version control.

```bash
uv run scripts/hx_downbeats.py --fair --split-flow song-cuts-final
uv run scripts/hx_embed_bars.py --downbeats fold --split-flow song-cuts-final --suffix _final
uv run scripts/hx_embed_matpac.py --downbeats fold --split-flow song-cuts-final --suffix _final
uv run scripts/hx_run.py --variant v10 --out .claude/hillclimb/song-cuts-final --cuts .claude/hillclimb/song-cuts/v10/cuts.json --emb-suffix _final
```

For the merge model use `v12/cuts.json`. For the baseline, use the baseline settings with `"refine": false` added and no `--emb-suffix`.
