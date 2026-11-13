# v12: learned keep/drop model on block-matching proposals, trained on the tuning songs' annotations (outer step)

Outer step 6, built by a subagent on train songs only: boosted trees on about 100 segment-relation features (MATPAC++, CLAP, PCA), replacing the median veto. Out-of-fold on train with nested selection: +0.7 ±1.3. The train column for this row is resubstitution (the model saw those annotations) and is not comparable; only the held-out column counts.
