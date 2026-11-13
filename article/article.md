# bar2vec

One vector per bar of music, used to cut a song into sections and name them.

Nobody tells bar2vec where the chorus is: an audio model that has never seen a labelled song section turns each bar into a list of numbers, and the sections fall out of asking which bars sound like which other bars. On 180 songs that played no part in tuning, its cuts score 72.7 against human annotations (boundary F-measure within three seconds), up from 35.2 for the fixed-window version it started as.

## Terms

This article uses the following definitions:

- A **bar** is one measure of music, about two seconds long. A beat tracker finds the bar lines, and each bar is one chunk of the song. The first version of the project chunked by fixed 10-second **windows** instead.
- An **embedding** (also called a vector) is the list of 3,840 numbers the audio model returns for one bar. Two bars that sound alike get similar lists.
- The **audio model** (MATPAC++) is pretrained and frozen: it was never trained on song structure, and nothing in it is changed here.
- The **similarity grid** compares every bar with every other bar. A dark square on the diagonal is a stretch of bars that sound the same throughout. A dark square off the diagonal is a stretch that comes back later.
- A **section** is one of those stretches, and a **cut** is where bar2vec ends one section and starts the next. An **annotated boundary** is where a human annotator did.
- The **cutter** picks the split whose squares hold together best in the similarity grid, then drops any cut where the bars on either side do not change.
- A **name** is the role a section plays: intro, verse, pre-chorus, chorus, post-chorus, bridge, outro or instrumental.
- The **namer** gives each section a name. The **rules** use the song's shape alone: the most repeated block is the chorus, a one-off late in the song is the bridge, and so on. The **classifier** uses the same structural features and is trained on 150 annotated songs. It is the one part that was told what a chorus is, and it is what the live demo uses.
- The **tuning songs** (153) are the ones settings were chosen on. The **held-out songs** (154) were drawn at random and only scored. The **locked songs** (180) were scored once, at the end.

## Results

Everything is measured on the Harmonix Set, a collection of pop, hip-hop, dance and country songs with human-annotated sections. Settings were chosen on 153 songs, checked on 154 held-out songs drawn at random, and confirmed once on 180 further songs that were locked away until the end.

Cutting, on the 180 locked songs. The score is boundary F-measure: how many cuts land within the tolerance of an annotated boundary, balanced against false cuts.

| Cutter | Within 3 s | Within 0.5 s |
|---|---|---|
| Starting point: a text-audio model on 10 s windows | 35.2 ±2.2 | 7.0 ±1.4 |
| bar2vec, no training | 72.7 ±2.6 | 53.9 ±3.4 |
| bar2vec plus a learned filter on its cuts | 75.4 ±2.4 | 55.3 ±3.3 |

Naming, on the same 180 locked songs, given the annotated boundaries. The score is macro-F1 over eight labels, so a rare label such as bridge counts as much as verse.

| Namer | Macro-F1 | Sections named right |
|---|---|---|
| Starting point: match each section to a text prompt | 9.3 ±2.0 | 14.0% |
| Rules, no training | 55.9 ±3.9 | 63.8% |
| Classifier, trained on 150 annotated songs | 65.9 ±3.7 | 74.7% |

The scores on the locked songs match the held-out scores from tuning to within a point and a half, so the rounds of tuning did not inflate them.

End to end, on the 154 held-out songs: bar2vec cuts the song and then names its own sections. Its sections do not line up one to one with the annotated ones, so the score is by time: each second of an annotated section is compared with the name bar2vec gives that second.

| Names from | On | Macro-F1 by time | Seconds named right |
|---|---|---|---|
| The annotation (best possible) | bar2vec's cuts | 89.5 ±1.6 | 91.0% |
| Rules | annotated boundaries | 54.9 ±4.3 | 65.7% |
| Rules | bar2vec's cuts | 44.2 ±3.4 | 54.9% |
| Classifier | annotated boundaries | 66.4 ±4.2 | 79.0% |
| Classifier | bar2vec's cuts | 53.1 ±2.9 | 67.9% |

The cuts cost the rules 10.7 points. Misplaced cuts alone account for 4.6 of them; the rest is the rules choosing different names when given the cuts. Naming is the weaker half: perfect names on the same cuts would score 89.5.

## How it started

The project began as moment search: cut every song in a playlist into overlapping 10-second windows, embed each window with a text-audio model, and let a text query return timestamps instead of whole songs. A side product was a song breakdown that cut each song where neighbouring windows changed the most.

That raised a question the prototype could not answer: were the cuts any good? Published section annotations exist, so the cutter could be scored against them and tuned.

## The cutter

A March 2026 paper (Marmoret et al.) had already shown that a frozen audio model can find section boundaries without training. Its settings were chosen on the same songs it reports. This project adds a random held-out split and a locked final test, and takes two ideas from the paper: embed bars instead of fixed windows, and its block-matching method for proposing cuts.

The cutter that came out of the climb works like this:

1. A beat tracker finds the bar lines.
2. The audio model turns each bar into one embedding.
3. Every bar is compared with every other bar, which gives the similarity grid.
4. Block matching proposes the split whose squares hold together best in the grid.
5. A change score, which compares the bars before and after each proposed cut, drops the cuts where nothing changes.

Most of the gain came from where a cut is allowed to fall. Moving from fixed windows to one embedding per bar added 17 points at three seconds and took the half-second score from 9.8 to 43.7. Swapping in a better audio model added about 2.

The beat tracker was trained on Harmonix, which would leak. Each song was therefore tracked by the cross-validation model that never saw it. Bars from human annotators, from the leaky tracker and from the fair tracker all gave the same score, so the gain is from bar alignment and not from the leak.


## The cutter climb

Twelve rounds were run, one change each. A change was kept only if both the tuning songs and the held-out songs improved by more than the noise between songs. The score is the three-second boundary score on the 154 held-out songs.

![Step graph of the held-out boundary score over the twelve rounds of the cutter climb. The kept rounds climb from 34.5 to 42.6 at round 1 and 50.8 at round 3, jump to 67.9 at round 7 with one vector per bar, then reach 70.0 and 72.2. Six rounds that were tried and not kept sit on or below the line. The supervised filter at round 12 is marked separately at 75.4.](./climb.svg)

| Round | Change | Held-out | Decision |
|---|---|---|---|
| 0 | A text-audio model (CLAP), 10 s windows every 5 s | 34.5 | starting point |
| 1 | Report each cut 2.5 s later, to the middle of the window overlap | 42.6 | kept |
| 2 | Lower the cut threshold | 43.4 | reverted, within noise |
| 3 | Place cuts between grid points | 50.8 | kept |
| 4 | A window every 2 s | 50.6 | reverted |
| 5 | One embedding per bar, bars from human annotators | 67.7 | upper-bound check |
| 6 | Bars from a tracker trained on these songs | 68.2 | leak check |
| 7 | Bars from tracker models that never saw the song | 67.9 | kept |
| 8 | Block matching alone | 67.8 | reverted |
| 9 | Block matching proposes, the change score vetoes | 70.0 | kept |
| 10 | MATPAC++ as the audio model in place of CLAP | 72.2 | kept |
| 11 | Three bars of context per embedding | 68.9 | reverted |
| 12 | A learned keep-or-drop filter on the proposed cuts | 75.4 | passes; supervised, reported separately |

Three things stand out:

- **The first 16 points cost nothing.** Rounds 1 and 3 only changed where a cut is reported, with the same embeddings.
- **Round 9 was a judgment call.** Its three-second gain (+2.1 ±2.2) sat just inside noise. It was kept for its half-second gain (+6.1 ±2.6).
- **Quick settings ran out twice.** After rounds 3 and 10, sweeps of every setting that needs no re-embedding stayed within about a point, which is why the later rounds are all changes of method.

The climb stopped by a rule agreed at the start: two expensive steps in a row that fail to beat noise. Round 11 and the beat-level placement work below were those two.

## Naming

Asking a text-audio model which section sounds like "the chorus of a song" scores at chance. A section's role is not a sound; it is a position in the song's structure. The rules that work use that structure:

- The first section is the intro and the last is the outro.
- Sections are grouped by similarity. The most repeated group is the chorus.
- A one-off section in the second half is the bridge.
- Short repeating sections beside a chorus are the pre-chorus or post-chorus.
- A section with far less singing than the rest of the song is instrumental.

Each rule was one round, and each cleared noise on the held-out songs.

![Step graph of held-out macro-F1 over the eight rounds of the naming climb. From 9.0 for text prompts, the five rules step up to 23.3, 34.6, 42.4, 53.0 and 56.0. The loudness round at 55.7 sits on the line and was not kept. The trained classifier is marked separately at 66.7, and the same classifier with twice the labels at 67.0 was not kept.](./names.svg)

| Round | Change | Held-out macro-F1 | Decision |
|---|---|---|---|
| 0 | Match each section to a text prompt | 9.0 | starting point; random labels score 11 |
| 1 | First section is intro, last is outro | 23.3 | kept |
| 2 | Most repeated group is the chorus | 34.6 | kept |
| 3 | One-off section in the second half is the bridge | 42.4 | kept |
| 4 | Short repeats beside the chorus are pre- or post-chorus | 53.0 | kept |
| 5 | Far less singing than the rest is instrumental | 56.0 | kept |
| 6 | Loudness helps pick the chorus group | 55.7 | reverted |
| 7 | Boosted trees over about 130 features per section | 66.7 | supervised, reported separately |
| 8 | The same with twice the labelled songs | 67.0 | not adopted, within noise |

The largest remaining loss for the rules is picking the wrong group as the chorus, which happens in 32 of the 153 tuning songs.

## What did not work

- **A bar-level transformer.** Each bar is a token and every bar attends to every other bar in the song. Cross-validated on 150 songs it reached 45 macro-F1, against 57 for the rules and 63 for boosted trees on the same songs. Attention did help: with it removed, sections named right fell from 65.5% to 57.2%. The model learned the common labels and missed the rare ones.
- **More labels.** 213 partly aligned Harmonix songs doubled the training data. The classifier gained in cross-validation and not on held-out songs (67.0 against 66.7, +0.3 ±2.4).
- **Loudness** as a hint for which group is the chorus gained on tuning songs and failed on held-out songs.
- **Beat-level placement.** 98% of annotated boundaries sit on a tracked beat but only 78% on a bar start, so moving cuts to the right beat should help. About 100 rule settings and a learned placer all failed to beat noise.

## Limits

- Only Harmonix has been measured. The naming rules encode Western pop song form.
- The end-to-end scores are on the 154 held-out songs. The 180 locked songs were scored with annotated boundaries only.
- The classifier was fitted on annotated boundaries and loses 13 points on bar2vec's cuts.
- The scores are not comparable with the paper's: the song subsets and audio differ.
- 487 of Harmonix's 912 songs could be matched to audio and aligned. The ones that aligned may differ from the ones that did not.
- Harmonix contains repeats and remixes. Nine groups of duplicates span the splits, which matters most for the two trained rows.
- Post-chorus and instrumental are the weakest labels (F1 19 and 27 for the rules).
- The output has been spot-checked by ear on some of the playlist, not reviewed in full.

## Next

- Fit the classifier on bar2vec's own cuts instead of annotated boundaries.
- Try other ways of grouping bars that sound alike (k-means, spectral clustering) for picking the chorus, the rules' largest remaining loss.
- Test whether a section's name can be read off its nearest neighbours in other songs.
- Return to learned attention between sections once there are more labelled songs.
- Compare songs section by section instead of averaging a whole song into one vector.
- Re-cut inside a long section at a finer scale.
- Test on music outside Harmonix.
