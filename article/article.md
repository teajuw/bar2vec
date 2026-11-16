# bar2vec

One vector per bar of music, used to cut a song into sections and name them.

Nobody tells bar2vec where the chorus is: an audio model that has never seen a labelled song section turns each bar into a list of numbers, and the sections fall out of asking which bars sound like which other bars. On 180 songs that played no part in tuning, its cuts score 72.7 against human annotations (boundary F-measure within three seconds), up from 35.2 for the fixed-window version it started as.

## Terms

This article uses the following definitions:

- A **bar** is one measure of music, about two seconds long. A beat tracker finds the bar lines.
- An **embedding** is the list of 3,840 numbers the audio model returns for one bar. Two bars that sound alike get similar lists.
- The **audio model** (MATPAC++ [1]) is pretrained and frozen. It was never trained on song structure.
- A **section** is a stretch of bars that sound alike, and a **cut** is where one section ends and the next starts.
- The **cutter** compares every bar with every other bar and cuts where the song changes. It uses no training.
- The **namer** gives each section a name: intro, verse, pre-chorus, chorus, post-chorus, bridge, outro or instrumental. The **rules** use the song's shape alone. The **classifier** is trained on 150 annotated songs; it is the one part that was told what a chorus is, and it is what the demo uses.
- A bar's **colour** in the demo is its embedding squeezed to three numbers, the three directions that vary most across the song, shown as red, green and blue. It keeps about 35% of the variation and is only for looking: the cutter uses all 3,840 numbers.

## Results

Everything is measured on the Harmonix Set [2], pop, hip-hop, dance and country songs with human-annotated sections. Settings were chosen on 153 songs, checked on 154 held-out songs drawn at random, and confirmed once on 180 songs locked away until the end.

Cutting, on the 180 locked songs. The score counts cuts that land near an annotated boundary, balanced against false cuts.

| Cutter | Within 3 s | Within 0.5 s |
|---|---|---|
| Starting point: 10-second windows | 35.2 ±2.2 | 7.0 ±1.4 |
| bar2vec, no training | 72.7 ±2.6 | 53.9 ±3.4 |

Naming, on the same songs, given the annotated boundaries. The score is macro-F1 over the eight names, so a rare name such as bridge counts as much as verse.

| Namer | Macro-F1 | Sections named right |
|---|---|---|
| Starting point: match each section to a text prompt | 9.3 ±2.0 | 14.0% |
| Rules, no training | 55.9 ±3.9 | 63.8% |
| Classifier | 65.9 ±3.7 | 74.7% |

End to end, bar2vec cuts a song and then names its own sections. Scored by time on the held-out songs, the classifier reaches 53.1 ±2.9, and perfect names on the same cuts would reach 89.5. Naming is the weaker half.

## How it works

The project began as search over 10-second windows embedded with a text-audio model (CLAP [3]). Its cuts could be scored against published annotations, so they were. A March 2026 paper had shown that a frozen audio model can find section boundaries without training [4]; this project takes its two main ideas, bars instead of windows and block matching, and adds a held-out split and a locked final test.

The cutter:

1. A beat tracker (Beat This! [5]) finds the bar lines.
2. The audio model turns each bar into one embedding.
3. Every bar is compared with every other bar.
4. Block matching proposes the split whose blocks of similar bars hold together best.
5. Any cut where the bars on either side do not change is dropped.

The namer's rules: the first section is the intro and the last is the outro, the most repeated group of sections is the chorus, a one-off late in the song is the bridge, short repeats beside a chorus are the pre-chorus or post-chorus, and a section with far less singing is instrumental. Asking a text-audio model which section sounds like a chorus scores at chance: a section's role is a position in the song, not a sound.

## How it got there

Twelve rounds, one change each. A change was kept only if the tuning songs and the held-out songs both improved by more than the noise between songs.

![Step graph of the held-out boundary score over the twelve rounds of the cutter climb. The kept rounds climb from 34.5 to 42.6 at round 1 and 50.8 at round 3, jump to 67.9 at round 7 with one vector per bar, then reach 70.0 and 72.2. Six rounds that were tried and not kept sit on or below the line. The supervised filter at round 12 is marked separately at 75.4.](./climb.svg)

- **Where a cut may fall mattered most.** Moving from fixed windows to one embedding per bar added 17 points. Swapping in a better audio model added about 2.
- **The first 16 points cost nothing.** Two early rounds only changed where a cut is reported, with the same embeddings.
- **The beat tracker did not leak.** It was trained on Harmonix, so each song was tracked by a model that never saw it. Bars from human annotators gave the same score.

Naming was climbed the same way, one rule a round. Every round of both climbs is in the repo's [process log](https://github.com/teajuw/bar2vec/blob/main/PROCESS.md).

## What did not work

- **A bar-level transformer** for names reached 45 macro-F1 against 57 for the rules on the same songs. It learned the common names and missed the rare ones.
- **Twice the labelled songs** did not help the classifier on held-out songs (+0.3 ±2.4).
- **Moving cuts to the nearest beat.** About 100 rule settings and a learned placer all failed to beat noise.
- **A learned filter on the cuts** does score higher (75.4) but is trained on one pop-heavy dataset, so the training-free cutter is the one reported.

## Limits

- Only Harmonix has been measured, and the naming rules encode Western pop song form.
- The classifier was fitted on annotated boundaries and loses 13 points on bar2vec's own cuts.
- 487 of Harmonix's 912 songs could be matched to audio. The ones that matched may differ from the ones that did not.
- Nobody has listened to the demo's songs in full. About a third of their section names are marked unsure.

## Next

- Fit the classifier on bar2vec's own cuts.
- Try other ways of grouping bars for picking the chorus, the rules' largest remaining loss.
- Test whether a section's name can be read off its nearest neighbours in other songs.
- Test on music outside Harmonix.

## Sources

1. [MATPAC++: Enhanced Masked Latent Prediction for Self-Supervised Audio Representation Learning](https://arxiv.org/abs/2508.12709), Quelennec et al., arXiv, August 2025. The audio model.
2. [The Harmonix Set](https://github.com/urinieto/harmonixset), Nieto et al., ISMIR 2019. The songs and their human-annotated sections.
3. [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687), Wu et al., arXiv, November 2022. CLAP, the text-audio model the project started with.
4. [Unsupervised Evaluation of Deep Audio Embeddings for Music Structure Analysis](https://arxiv.org/abs/2603.27218), Marmoret, arXiv, March 2026. The paper the bar embeddings and block matching come from.
5. [Beat This! Accurate beat tracking without DBN postprocessing](https://arxiv.org/abs/2407.21658), Foscarin et al., arXiv, July 2024. The beat tracker.
