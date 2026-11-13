# v8: trained namer refitted with 213 extra, partly aligned Harmonix songs (training only)

Outer step 3: more labels. 344 downloaded Harmonix songs failed the alignment bar for scoring; 213 have at least 60 s that lines up with the annotated recording and are no repeat of a held-out or final song (scripts/hx_tracks_extra.py). A section keeps its label when at least 75% of its bars are in aligned audio: 1,509 extra labelled sections against 1,519. Same untuned classifier as v7. Out-of-fold on the 150 train songs: 66.5 against 62.6, +3.9 ±2.9. Train column is resubstitution.
