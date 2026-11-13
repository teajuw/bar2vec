# v6: chorus pick: repeat count plus 2 x the group's loudness relative to the rest of the song

Outer step 1 (new signal: per-bar mean log-mel energy, scripts/hx_loud_bars.py). Analyzer: the wrong group is picked as chorus in 32 of 153 train songs, 16 on count ties. Hypothesis: the chorus is the loudest of the often-repeated groups. Loudness = section mean of per-bar mean log-mel energy, z-scored across the song's sections. Train sweep of the weight 1 / 1.5 / 2 / 3 / 4 / 8: 57.8 / 58.3 / 58.2 / 58.3 / 58.3 / 57.0 against 56.4. Plain decibel level was weaker (57.2 at best).
