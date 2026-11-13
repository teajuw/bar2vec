# Brief: music as an AI domain

**What:** I am applying standard AI techniques (embeddings, chunking, search, and reading model outputs back into English) to a personal music library, starting with a working prototype that searches songs by moment instead of by whole song.

## So what

### One building block opens up several features

Music apps treat a song as one unit: one title, one genre, one "vibe". The prototype cuts every song into overlapping ten-second windows and turns each window into an embedding, a list of numbers that captures how that stretch of audio sounds. The same model can turn a text description into the same kind of numbers, so text and audio can be compared directly.

That one step, done once per song, is the foundation for everything else:

| Next step | What it gives a listener | What it needs beyond the windows |
|---|---|---|
| Moment search by text (built) | "Find the slow dreamy part", answered with a song and a timestamp | Nothing |
| More like this moment (built) | Pick ten seconds you like, get similar moments in other songs | Nothing |
| Song breakdown | A timeline of each song's sections, labelled | Comparing a song's windows with each other; no new model |
| Plain-English description | A short written summary of how a song unfolds | A language model to write it from the breakdown |
| Readable explanations | Why the system matched or chose something, in words | A list of descriptions to compare each window against |
| Set-building agent | An assistant that assembles a playlist or DJ set and can explain each pick | All of the above |

The last two rows are the reason this is more than a search feature. An AI agent that picks music from numbers alone cannot be inspected. If each choice can be read back as "slow, heavy reverb, no drums", a person can check its reasoning, which is the same problem as making any AI agent's decisions observable.

### Why music

The techniques are the ones already used for documents and images. Splitting a song into windows is the same move as splitting a long document into passages for search, and the audio models are image models applied to a picture of the sound. Music is a domain where these methods are well understood but few consumer features use them below the level of a whole song. Spotify's current AI features (prompted playlists, an AI DJ, automatic transitions) all operate on whole songs.

### What exists today

- A library of 81 songs cut into 3,203 windows, embedded with two different audio-text models (CLAP and MuQ-MuLan).
- Text search and similar-moment search from the command line, plus web pages where each match can be played from its timestamp.
- A small test: 18 written queries, each with the song it should find.

### What the test showed

- **Using every window beats sampling one.** My earlier project embedded a single random ten-second sample per song. That scored lower than best-window search on all 20 random runs, for both models, and its results changed with which ten seconds were sampled.
- **For finding the right song, averaging a song's windows did as well as or better than taking its best window** with one model (right song first for 67% of queries versus 56%). Most test queries describe a whole song, which favours averaging.
- **The one query about a specific section favoured moment search:** the expected song ranked 5th by best window and 23rd by average.
- **The two models often disagree** about which moments sound alike.

### What is not proven yet

- **Nobody has listened to the matches.** The test checks that the right song comes back, not that the timestamp points at the right part.
- **The 18 test queries are drafts** written without hearing the songs, and at least one expected answer looks wrong.
- **The sample is small.** 81 songs from one playlist and 18 queries cannot separate the two models or settle best-window versus average.
- **Only the first two rows of the table are built.** The breakdown, descriptions, explanations and agent are plans.
- **This is a personal prototype.** The audio is for private experimentation, and one model's licence is non-commercial.

The next step that would turn this from a demo into evidence is a listening pass, followed by test queries that name specific moments, since that is the case moment search exists for.
