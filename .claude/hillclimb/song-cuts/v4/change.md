# v4: window every 2 s instead of 5 s

Hypothesis: the 5 s grid limits timing, so a 2 s grid should place cuts more precisely. Offset moves with the spacing (half the overlap = 4 s by geometry). Train sweep of offset_s at 2 s spacing: 3 -> 46.5, 3.5 -> 47.7, 4 -> 49.3, 4.5 -> 50.8, 5 -> 50.6. Candidate 4.5 (train best). Note before running: every train value is below v3's 52.8, so this is expected to fail; run for the record.
