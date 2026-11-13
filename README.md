# bar2vec

Music as a domain for standard AI techniques: one embedding per bar or window instead of one per song. This repo holds a moment-search prototype (below) and a training-free song-section cutter tuned on the Harmonix Set; the cutter work and its results are in [PROCESS.md](PROCESS.md). The project was called `moment-search` until 2026-11-13.

## Moment search

Search a music library by moment instead of by song: every track is cut into overlapping ten-second windows, each window gets its own audio-text embedding, and a text query returns matching windows with timestamps.

Prototype built 2026-11-11 on the 82-track playlist "stimmy check?" (81 unique songs, 3,203 windows).

## Results so far

18 draft test queries, each with one or more expected tracks. The score is the rank of the best expected track (MRR = mean reciprocal rank; higher is better). This measures whether the right *song* comes back, not whether the *timestamp* is right; timestamps need listening.

| Ranking method | CLAP (music checkpoint) | MuQ-MuLan |
|---|---|---|
| Moment (track scored by its best window) | MRR 0.653, hit@1 56%, hit@5 78% | MRR 0.627, hit@1 50%, hit@5 72% |
| Whole-track mean of windows | MRR 0.738, hit@1 67%, hit@5 78% | MRR 0.579, hit@1 44%, hit@5 78% |
| One random crop (what `vibe` does today) | MRR 0.540 (0.43–0.64 by seed) | MRR 0.512 (0.41–0.62 by seed) |

What this does and does not show:

- **Windowing beats vibe's current approach.** laion-clap with fusion off embeds one random ten-second crop per file (confirmed in the library source), so results depend on which ten seconds it happens to pick. Any use of all windows is better and stable.
- **For finding the right song, averaging the windows is as good as or better than taking the best window** with CLAP. Most draft queries describe a whole song ("country song with twangy guitar"), which favours averaging.
- **The one clearly moment-specific query favours moment search:** "slow dreamy psychedelic section" ranks the expected track 5th by best window versus 23rd by mean (CLAP).
- **No clear winner between models.** 18 queries is too few to separate them; each wins on different queries.
- **The queries are drafts** written without listening, and at least one expectation looks wrong (the track assumed to be a spoken skit does not behave like one).

## Layout

| Path | What it is |
|---|---|
| `data/playlist.json` | Track list read from the playlist's public embed page |
| `data/downloads.json` | Which video each track was matched to, with duration difference |
| `data/tracks.json` | Tracks in index order |
| `data/queries.json` | Draft test queries and expected tracks. Edit this |
| `data/clap_w10_h5.npz`, `data/muq_w10_h5.npz` | Window embeddings (10 s window, 5 s hop) |
| `data/eval_*.json` | Per-query results behind the listening pages |
| `results_clap_w10_h5.html`, `results_muq_w10_h5.html` | Listening pages: press play to hear each matched moment |
| `audio/`, `models/` | Downloaded audio and the CLAP checkpoint (gitignored) |

## Commands

```bash
uv run scripts/download.py            # fetch audio for data/playlist.json
uv run scripts/embed_clap.py          # --window 10 --hop 5 by default
uv run scripts/embed_muq.py           # same windows with MuQ-MuLan
uv run scripts/search.py "a heavy drop" --index clap_w10_h5
uv run scripts/evaluate.py --index clap_w10_h5
uv run scripts/make_page.py --index clap_w10_h5
uv run scripts/similar.py "self care" 4:30     # moments that sound like this one
uv run scripts/make_similar_page.py            # similar.html
uv run scripts/readout.py "self care" 4:30     # read a moment (or a whole song) back into English
uv run scripts/roundtrip.py                    # describe a window, search with the description
uv run scripts/make_readout_page.py            # readout.html
uv run scripts/breakdown.py --index clap_w10_h5 --show "self care"   # sections for every song
uv run scripts/make_breakdown_page.py          # breakdown.html (run breakdown.py for both indexes first)
```

## Song breakdown

Each song is cut where the windows before a point stop resembling the windows after it, sections that resemble each other get the same letter, and each section is described with the readout. Nothing is trained. Songs get a median of 8 sections (CLAP) or 7 (MuQ-MuLan).

Checked without listening: 79% of CLAP's boundaries have a MuQ-MuLan boundary within five seconds in the same song, against 48% when paired with a different song. Boundaries are accurate to about five seconds at best. The cut threshold and the "same letter" threshold were set once and not tuned; whether the sections match what a listener hears is unverified.

## Readout (embeddings back to English)

Each window is compared with the 128 descriptions in `data/vocab.json`, and each description is scored relative to the rest of the library. Checked two ways, neither by ear:

- **Known songs:** genre is mostly right for 14 songs whose genre is not in doubt (Christmas, country, dance, piano ballad, hip hop, rock). The vocals descriptions are unreliable (for example "rapping" on a folk song).
- **Round trip on 300 random windows** (describe with four descriptions, search with that text; chance is 1% song-first, 6% top-5):

| Describe with | Search with | Right song first | Right song in top 5 |
|---|---|---|---|
| CLAP | CLAP | 14% | 51% |
| MuQ-MuLan | MuQ-MuLan | 29% | 74% |
| CLAP | MuQ-MuLan | 9% | 28% |
| MuQ-MuLan | CLAP | 10% | 44% |

The cross-model rows are the meaningful ones: a description written by one model leads the other, separately trained model back to the right song far more often than chance.

## Notes

- `transformers` is pinned below 5 because MuQ-MuLan fails on 5.x.
- One download is flagged for a duration mismatch of about nine seconds (track 27), so it may be a different version.
- MuQ-MuLan weights are CC-BY-NC (non-commercial).
- Parked settings to score later: window length, hop (2 s and 10 s), section boundaries from a structure model instead of fixed windows.
