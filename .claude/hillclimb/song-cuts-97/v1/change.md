# v1: report each cut 2.5 s later

Hypothesis: a cut was reported at the start of the window after the change, but the change sits in the middle of the overlap between the windows before and after, which is (window - hop) / 2 = 2.5 s later. Train sweep of offset_s: 0 -> 37.3, 1.25 -> 41.1, 2.5 -> 44.4, 3.75 -> 43.0, 5 -> 37.7. Candidate 2.5 (train best, and the value the geometry predicts).
