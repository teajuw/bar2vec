# Other tools

## Big streaming services

| Service | Feature | What it does | Notes |
|---|---|---|---|
| Apple Music | AutoMix (iOS 26) | Automatic DJ-style transitions using beat matching and time stretching | Replaces crossfade; works better for some genres; some users say it cuts tracks short; AirPlay support added in iOS 26.1 |
| Apple Music | Playlist Playground | Prompt-to-playlist | Appeared in the iOS 26.4 beta, February 2026 |
| YouTube Music | AI playlists | Prompt-to-playlist | Rolled out February 2026, Premium only; the saved playlist does not keep the prompt |
| Amazon Music | Maestro | Prompt-to-playlist, accepts emoji | Beta since 2024; reviewers found it solid for mainstream requests, thinner for niche ones |

Takeaway: every major service now has prompt-to-playlist, and two have automatic transitions. Neither is a differentiator any more.

## Spotify inside ChatGPT (October 2025)

- Mention Spotify in a ChatGPT conversation, connect your account, and ask for songs, playlists or podcast episodes by mood, theme or topic.
- Free users get existing editorial playlists; Premium users get custom playlists from detailed prompts.
- It can use the surrounding conversation (for example a road-trip plan) as context.
- Reported as available in English in 145 countries.

## DIY and research projects

No ready-made third-party app was found that searches a Spotify library by sound. What turned up instead:

- **A developer write-up doing the same thing as `vibe`:** 30-second iTunes preview clips, LAION CLAP embeddings, pgvector for similarity. Their lesson: audio similarity captures sonic texture but not intent, so they added LLM-generated tags on top.
- **CLAP** remains the standard choice for matching text descriptions to audio.
- **Spotify's own research** publishes on semantic IDs and LLM-based recommendation, which suggests their internal models are well ahead of what the public features expose.

Gap in this pass: I did not survey smaller consumer apps (playlist analysers, stats sites, DJ software such as rekordbox or djay). Those are the likeliest place for overlap with the transition and sequencing ideas.
