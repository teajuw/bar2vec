# Spotify API constraints for a third-party app

This section was checked against Spotify's developer pages (search summaries of them, not a full read). Confirm the details there before building.

## Since November 2024

Unavailable to new apps:

- Audio features (BPM, key, energy, danceability) and audio analysis
- Recommendations and related artists
- 30-second audio previews

Effect: `Soupify`'s approach (rank tracks by Spotify's audio features) cannot be rebuilt on a new app.

## Since February 2026 (development mode)

- The app owner needs a Premium account; if it lapses, the app stops working.
- Each app is capped at **five authorized users**.
- One development-mode client ID per developer at announcement. One guide says this was raised to 25 in July 2026; unconfirmed.
- Search results: maximum 10 per request (was 50), default 5.
- Removed: batch fetches (`GET /tracks`, `/albums`, `/artists`), browse and new releases, artist top tracks, other users' profiles and playlists.
- Library endpoints consolidated under `/me/library`; playlist `/tracks` endpoints renamed to `/items`.
- The endpoint removals were postponed for existing apps, but the Premium requirement and user cap applied on schedule.
- Apps with Extended Quota were not affected, but getting that status has strict requirements.

## What still works

- Reading the user's own saved tracks and playlists
- Creating and editing the user's own playlists
- Playback control (Premium)
- Search, with the smaller result limit

## What this means for `vibe`

- **Still feasible as a personal tool:** read your library, get audio elsewhere, compute your own embeddings, write playlists back.
- **Not feasible as a public app:** five users maximum without Extended Quota.
- **Audio source is the weak point:** Spotify provides none, so `vibe` uses yt-dlp. That is fine for a personal demo and a problem for anything distributed.
- **Code check needed:** `vibe` was last pushed in March 2026, around the migration date, so its sync code may use renamed or removed endpoints.
