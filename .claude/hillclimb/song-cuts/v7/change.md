# v7: one embedding per bar, bars from a beat tracker that never trained on the song

Fair version of v6. Harmonix is in the Beat This! tracker's training data (confirmed from its published 8-fold split), so v6's bars came from a model that had learned from human downbeat annotations on these songs. Here each song is tracked by the cross-validation model whose training left that song out (data/beat_this_harmonix_folds.tsv). Everything else is identical to v6: one CLAP embedding per bar, round-3 cutter with sub-grid placement and timing offset off, settings unchanged.
