# Ideas: standard AI techniques applied to a personal music library

Saved 2026-11-11. Framing: treat music as "documents in another modality." Most standard AI techniques have a direct analogue, and few have been applied to a personal library.

| Idea | Normal AI thing it's based on | What the user gets |
|---|---|---|
| **Moment search** | Chunking in RAG | Embed each section of a song instead of the whole track, so you can search for "the drop" or "a bridge like this one." |
| **Stem-level similarity** | Multi-vector retrieval | Separate drums, bass and vocals, embed each, then ask for "these drums with a different kind of vocal." |
| **Taste dials** | Sparse autoencoders (interpretability) | Decompose the embeddings into nameable features such as "reverb-heavy" or "breathy vocal," and expose them as sliders. |
| **Playlists as a benchmark** | Retrieval evals | Your own playlists become ground truth for "these belong together," so every model or prompt change gets a score. |
| **Similar to me** | Fine-tuning embeddings (metric learning) | A small adapter trained on your playlists and skips, so "similar" means how you group music, not how the model does. |
| **Will I save this?** | Cold-start classification | Predict from audio alone whether you'd keep a brand-new track with zero plays, where play-data recommenders are weakest. |
| **Liner notes for my library** | Captioning plus RAG | Generate a grounded description of each track from audio, lyrics and features, then answer "why do I like this?" |
| **Set-building agent** | Tool-using agents | An agent with tools for sound search, tempo and key extraction, and playlist writing plans a set and critiques its own draft. |

## How they relate to the earlier one-sentence ideas

- **Lyrics versus sound** is cross-modal disagreement; the liner-notes idea generalises it.
- **Start here, land there** becomes the set-building agent's main task, and moment search improves where it joins tracks.
- **Find the saved song that sounds like this** gets sharper with stems and with the "similar to me" adapter.

## Suggested order

1. **Playlists as a benchmark.** Cheap, fits an eval-driven way of working, and makes every other idea measurable.
2. **Moment search.** A small change to `vibe`'s pipeline with an obvious demo.
3. **Taste dials.** Most novel, but a few thousand songs is too little data to train a sparse autoencoder; it would need a public corpus. Prior work on audio models not yet checked.

The set-building agent is the most crowded at the surface, since Talk to Spotify looks similar; its edge would be the tools underneath, not the chat.

## Favourites flagged for a closer look

- Set-building agent
- Moment search
