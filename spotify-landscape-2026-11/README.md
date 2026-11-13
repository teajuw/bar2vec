# Spotify landscape — quick research pass (2026-11-11)

Purpose: find out what Spotify and the other big streaming apps already do, so new music-project ideas (building on `vibe`, `Soupify`, `NextUp`) aren't pitched against features that already exist.

## Headline findings

1. **Prompt-to-playlist is now table stakes.** Spotify has Prompted Playlists (uses your full listening history, explains each pick, can auto-refresh), and Apple Music, YouTube Music and Amazon Music all have an equivalent.
2. **Spotify has a conversational assistant.** "Talk to Spotify" (beta since July 2026) takes typed or spoken requests, keeps context across turns, and answers questions about your listening history.
3. **Spotify already exposes BPM and key.** The Mix feature shows tempo and Camelot key for every track in a playlist, offers transition presets, and Smart Reorder sorts a playlist by BPM and key in one tap. Apple Music's AutoMix beat-matches automatically.
4. **Nothing found that searches your own library by how songs sound**, compares lyrics against sound, or builds a path between two songs. These remain open.
5. **The Spotify API is much tighter than when `vibe` was written.** Development-mode apps now need a Premium owner and are capped at five authorized users; audio features, recommendations and previews stay unavailable to new apps.

## Files

| File | Contents |
|---|---|
| [01-spotify-features.md](01-spotify-features.md) | Each Spotify feature: what it does, who gets it, what it does not do |
| [02-competitors.md](02-competitors.md) | Apple Music, YouTube Music, Amazon Music, ChatGPT integration, DIY projects |
| [03-api-constraints.md](03-api-constraints.md) | What a third-party app can and cannot do with the Spotify API today |
| [04-implications-for-ideas.md](04-implications-for-ideas.md) | The six one-sentence ideas re-scored against these findings |
| [05-ai-technique-ideas.md](05-ai-technique-ideas.md) | Eight ideas, each tied to the AI technique it is based on |
| [06-embedding-models.md](06-embedding-models.md) | CLAP compared with newer audio-text models |
| [07-vision-recipes-for-music.md](07-vision-recipes-for-music.md) | Computer-vision recipes and their music equivalents |
| [sources.md](sources.md) | Every link used, grouped by topic |

## How this was done, and its limits

- About fifteen web searches on 2026-11-11; I read search summaries, not every article in full.
- Most detail comes from tech press and blogs. Only the API section was checked against Spotify's own developer pages.
- Many features are betas with staged rollouts, so availability by country and tier may have changed since the articles were written.
- Not researched: Spotify's lyrics features, Wrapped-style listening stats, and smaller third-party apps (playlist analysers, DJ software). "Nothing found" in these notes means not found in this pass, not proven absent.
- Nothing here was tested hands-on in the Spotify app.
