# Song-cut climb: status

Target: f3_trim, 0-100. 49 train / 48 test songs (random, balanced by genre). Deltas are paired per song versus baseline.

| round | change | test | train | test delta | train delta | half-second (test) | decision |
|---|---|---|---|---|---|---|---|
| 0 | baseline: pipeline as built (CLAP, 10 s windows every 5 s) | 39.3 | 37.3 | | | 7.0 | |
| 1 | report each cut 2.5 s later | 42.5 | 44.4 | +3.2 ±4.7 (within noise) | +7.1 ±5.7 | 10.3 | inconclusive, reverted |

Best so far by the keep rule: baseline. Round 1 moved cut times to the middle of the window overlap, a correction that follows from the window geometry. It rose on train beyond noise and rose on test within noise, so it is not proven on held-out songs; across all 97 songs the paired gain is +5.2 ±3.7.

Inner-loop sweeps on train only (with the 2.5 s correction applied) show the remaining quick settings sit at or near their best values: change-score span 15 s is best (10 s: -3.8, 20 s: -4.5); cut threshold -0.5 gives +1.6; minimum gap 10 s gives +0.4; MuQ-MuLan alone is -6.0 and both models combined -5.6 versus CLAP alone. None of these is large enough to clear the held-out noise of about ±4.7, so the inner loop is saturated at this sample size. The binding limit is the number of songs, not the settings.
