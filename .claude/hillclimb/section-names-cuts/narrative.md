# Naming on bar2vec's own cuts

| round | change | held-out macro-F1 by time | tuning, out-of-fold | held-out seconds right | kept |
|---|---|---|---|---|---|
| 0 | baseline: classifier fitted on annotated sections (v7) | 53.1 ±2.9 | 50.1 ±3.0 | 67.9 | |
| 1 | fitted on annotated sections plus the cutter's own sections | 53.2 (+0.1 ±2.2) | 52.8 (+2.7 ±2.2) | 68.4 (+0.5 ±1.5) | no |
| 2 | per-bar votes added to the section classifier's inputs | 55.8 ±3.2 (+2.7 ±2.3) | 53.8 (+3.7 ±2.4) | 69.7 (+1.8 ±1.8) | yes |

Best so far: round 2. A second classifier names every bar from inputs that do not depend on where the cuts are (place in the song, how the bar repeats, how sung and loud it is, its vectors, each also averaged over nearby bars). Each section's bars vote, and the vote, with the neighbouring sections' votes, is added to the section classifier's inputs. Held-out +2.7 ±2.3 and tuning +3.7 ±2.4: both beyond noise, by a small margin, and seconds right did not fall.

**Versus baseline.** 53.1 to 55.8 on held-out. The ceiling on the same cuts is 89.5, so most of the gap remains.

**Why trust this.** The baseline reproduces the earlier measurement exactly. Fits are deterministic. One held-out look per round. The bar votes the section classifier is fitted on come from bar models that never saw the song. Caution: the margin over noise is 0.4 points, and the candidate was the better of two tried on the tuning songs.

**What else was tried.** Round 1 (fitting on the cutter's sections) failed on held-out. The bar votes alone, with no section classifier: tuning +2.1 ±2.5 macro-F1, +3.3 ±1.8 seconds; not scored on held-out.

**What the gap is made of** (tuning songs, baseline): 61.8 on annotated edges; 58.0 with those names carried onto the cuts (misplaced edges alone); 50.1 when the classifier names the cut sections itself.

**Not done.** v2 is not wired into `names.py`, the saved model or the demo. The 180 final songs have not been scored end to end.
