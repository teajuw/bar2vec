# The six ideas, re-scored

Status key: **Open** = nothing found that does it. **Narrowed** = part of it exists; a smaller claim survives. **Covered** = an existing feature does most of it.

| # | One sentence | Status | What exists | What is left |
|---|---|---|---|---|
| 1 | "Find the song I already saved that sounds like this." | Narrowed | Liked Songs has preset mood and genre chips; Prompted Playlists and Talk to Spotify use your listening history | Search by actual sound, seeded by a song rather than a description, restricted to what you saved |
| 2 | "Show me songs that sound happy but have sad lyrics." | Open | Nothing found | All of it. Caveat: Spotify's lyrics features were not researched in this pass |
| 3 | "Build me 45 minutes that starts at this song and lands at that one." | Narrowed | Smart Reorder sorts an existing playlist by BPM and key | Choosing the songs in between and shaping the arc; Smart Reorder only reorders what you give it |
| 4 | "Tell me which of my songs mix out of this one, and why." | Covered | Mix shows BPM and Camelot key per track and advises matching them; Smart Reorder; Apple AutoMix | Only the teaching angle and suggesting candidates from the whole library |
| 5 | "Show me a map of my library and where the holes are." | Open | Taste Profile beta lists listening data; SongDNA maps credits and samples | A map by sound. Neither existing feature is one |
| 6 | "More like this, but slower / darker / without vocals." | Covered | Talk to Spotify keeps context so you can refine in conversation; Prompted Playlists let you edit the prompt | Only precision: steering by a measured direction instead of re-prompting. Hard to show the difference in a demo |

## What changed from the earlier picks

- **Idea 2 (lyrics versus sound) holds up best.** Nothing found touches it, it is one sentence, and it needs both of `vibe`'s encoders.
- **Idea 3 (start here, land there) survives in a sharper form.** Pitch it as "Spotify can reorder a playlist; this picks the songs that get you from A to B."
- **Idea 4 (transitions) drops.** This was `NextUp`'s core, and Spotify now shows the same BPM and key data to every Premium user. What remains of `NextUp` is the education angle, which is a thinner pitch.
- **Idea 6 drops.** Conversational refinement is exactly what Talk to Spotify does.
- **Idea 5 is open but is a visualisation, not a capability.** Better as a supporting screen than as the headline.

## Suggested checks before committing

1. Try ideas 1 and 2 as prompts in Prompted Playlists and Talk to Spotify. If "songs from my likes that sound upbeat but have sad lyrics" returns good results, idea 2 is weaker than it looks here.
2. Look at DJ software and playlist-analysis apps for idea 3; they were not covered.
3. Run `vibe`'s sync against the current API to see what broke in the 2026 migration.
