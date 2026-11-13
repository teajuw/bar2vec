# v2: lower the cut threshold (k 0 to -1)

Hypothesis: the pipeline makes too few cuts (7.4 per song against about 9 annotated inner boundaries), so real boundaries with a modest change score are being discarded. Train sweep of k: -1.5 -> 48.3, -1 -> 48.5, -0.5 -> 47.9, 0 -> 46.2, 0.5 -> 43.6. Candidate -1 (train best).
