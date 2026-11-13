# bar2vec

One vector per bar of music, used to cut a song into sections and name them.

bar2vec finds where a song's sections begin and labels each one (verse, chorus, bridge) using frozen audio embeddings and no training. On 180 songs that played no part in tuning, its cuts score 72.7 against human annotations (boundary F-measure within three seconds), up from 35.2 for the fixed-window version it started as.

## Results

Everything is measured on the Harmonix Set, a collection of pop, hip-hop, dance and country songs with human-annotated sections. Settings were chosen on 153 songs, checked on 154 held-out songs drawn at random, and confirmed once on 180 further songs that were locked away until the end.

Cutting, on the 180 locked songs. The score is boundary F-measure: how many cuts land within the tolerance of an annotated boundary, balanced against false cuts.

| Cutter | Within 3 s | Within 0.5 s |
|---|---|---|
| Starting point: text-audio embeddings on 10 s windows | 35.2 ±2.2 | 7.0 ±1.4 |
| bar2vec, no training | 72.7 ±2.6 | 53.9 ±3.4 |
| bar2vec plus a learned filter on its cuts | 75.4 ±2.4 | 55.3 ±3.3 |

Naming, on the same 180 songs, given the annotated boundaries. The score is macro-F1 over eight labels, so a rare label such as bridge counts as much as verse.

| Namer | Macro-F1 | Sections named right |
|---|---|---|
| Starting point: match each section to a text prompt | 9.3 ±2.0 | 14.0% |
| Structural rules, no training | 55.9 ±3.9 | 63.8% |
| Classifier trained on 150 annotated songs | 65.9 ±3.7 | 74.7% |

The scores on the locked songs match the held-out scores from tuning to within a point and a half, so the rounds of tuning did not inflate them.

## How it started

The project began as moment search: cut every song in a playlist into overlapping 10-second windows, embed each window with a text-audio model, and let a text query return timestamps instead of whole songs. A side product was a song breakdown that cut each song where neighbouring windows changed the most.

That raised a question the prototype could not answer: were the cuts any good? Published section annotations exist, so the cutter could be scored against them and tuned.

## The cutter

A March 2026 paper (Marmoret et al.) had already shown that frozen audio embeddings can find section boundaries without training. Its settings were chosen on the same songs it reports. This project adds a random held-out split and a locked final test, and takes two ideas from the paper: embed bars instead of fixed windows, and its block-matching method for proposing cuts.

The cutter that came out of the climb works like this:

1. A beat tracker finds the bar lines.
2. An audio encoder (MATPAC++) turns each bar into one vector.
3. Block matching proposes the segmentation whose sections hold together best in the bar-by-bar similarity matrix.
4. A change score, which compares the bars before and after each proposed cut, drops the cuts where nothing changes.

Most of the gain came from where a cut is allowed to fall. Moving from fixed windows to one embedding per bar added 17 points at three seconds and took the half-second score from 9.8 to 43.7. Swapping in a better encoder added about 2.

The beat tracker was trained on Harmonix, which would leak. Each song was therefore tracked by the cross-validation model that never saw it. Bars from human annotators, from the leaky tracker and from the fair tracker all gave the same score, so the gain is from bar alignment and not from the leak.


## The cutter climb

Twelve rounds were run, one change each. A change was kept only if both the tuning songs and the held-out songs improved by more than the noise between songs. The score is the three-second boundary score on the 154 held-out songs.

![Bar chart of the held-out boundary score for all twelve rounds of the cutter climb. Kept rounds rise from 34.5 to 42.6, 50.8, 67.9, 70.0 and 72.2, with the supervised filter at 75.4. Reverted rounds and checks sit level with the round before them, and round 11 drops to 68.9.](./climb.svg)

| Round | Change | Held-out | Decision |
|---|---|---|---|
| 0 | Text-audio embeddings (CLAP), 10 s windows every 5 s | 34.5 | starting point |
| 1 | Report each cut 2.5 s later, to the middle of the window overlap | 42.6 | kept |
| 2 | Lower the cut threshold | 43.4 | reverted, within noise |
| 3 | Place cuts between grid points | 50.8 | kept |
| 4 | A window every 2 s | 50.6 | reverted |
| 5 | One embedding per bar, bars from human annotators | 67.7 | upper-bound check |
| 6 | Bars from a tracker trained on these songs | 68.2 | leak check |
| 7 | Bars from tracker models that never saw the song | 67.9 | kept |
| 8 | Block matching alone | 67.8 | reverted |
| 9 | Block matching proposes, the change score vetoes | 70.0 | kept |
| 10 | MATPAC++ encoder in place of CLAP | 72.2 | kept |
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

![Bar chart of held-out macro-F1 for the naming climb. The text-prompt starting point scores 9.0. The five rules raise it to 23.3, 34.6, 42.4, 53.0 and 56.0. A loudness rule at 55.7 is reverted, a trained classifier reaches 66.7, and the same classifier with twice the labels scores 67.0 and is not adopted.](./names.svg)

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
- **More labels.** 213 partly aligned Harmonix songs doubled the training data. The trained classifier gained in cross-validation and not on held-out songs (67.0 against 66.7, +0.3 ±2.4).
- **Loudness** as a hint for which group is the chorus gained on tuning songs and failed on held-out songs.
- **Beat-level placement.** 98% of annotated boundaries sit on a tracked beat but only 78% on a bar start, so moving cuts to the right beat should help. About 100 rule settings and a learned placer all failed to beat noise.

## Limits

- Only Harmonix has been measured. The naming rules encode Western pop song form.
- Names were scored on human-annotated boundaries. Names on bar2vec's own cuts are on the playlist page but have not been scored.
- The scores are not comparable with the paper's: the song subsets and audio differ.
- 487 of Harmonix's 912 songs could be matched to audio and aligned. The ones that aligned may differ from the ones that did not.
- Harmonix contains repeats and remixes. Nine groups of duplicates span the splits, which matters most for the two trained rows.
- Post-chorus and instrumental are the weakest labels (F1 19 and 27 for the rules).
- The output has been spot-checked by ear on some of the playlist, not reviewed in full.

## Next

- Score the names end to end, on bar2vec's own cuts.
- Compare songs section by section instead of averaging a whole song into one vector.
- Re-cut inside a long section at a finer scale.
- Test on music outside Harmonix.
