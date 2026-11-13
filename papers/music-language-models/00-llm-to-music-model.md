# From LLMs to music language models

Saved 2026-11-12 from the conversation. The question was whether "tokenize, self-attend, predict the next token" carries over from English to sound.

## Short answer

Yes. It is how a large share of music generators work. There is no single acronym like "LLM"; the usual terms are **music language model** or **audio language model**. Known examples: Jukebox (OpenAI, 2020), AudioLM and MusicLM (Google, 2022–23), MusicGen (Meta, 2023).

## The mapping

| | LLM | Music language model |
|---|---|---|
| Raw input | text | waveform |
| Tokenizer | splits text into word pieces from a fixed vocabulary | a neural audio codec (EnCodec, SoundStream, DAC) compresses audio into a stream of codes from a learned codebook |
| Model | transformer with self-attention | the same |
| Training task | predict the next token | predict the next audio token |
| Output | tokens → text | tokens → the codec's decoder → sound |

## The one correction

These models don't usually tokenize spectrograms. Spectrogram patches are continuous numbers, and "predict the next token" needs a finite vocabulary to choose from. The codec provides one by snapping each short slice of audio (about 1/50 of a second) to the nearest entries in a codebook.

## What is harder than text

- **Length.** At about 50 slices per second with several codes each, a three-minute song is tens of thousands of tokens. Keeping verse-chorus structure coherent over that span is hard.
- **Stacked codes.** Each slice needs several codes, coarse to fine, to sound good, so the model predicts several streams at once.
- **Steering.** Text prompts come in through a text-audio model such as CLAP or MuLan, the same kind this project uses for search.

## The other family

Diffusion models (Stable Audio, Riffusion) skip tokens. They start from noise and refine a spectrogram or a compressed version of the audio, the way image generators do.

## How it connects to this project

- **Our encoders are the "BERT" side of the same idea.** MATPAC++ and MuQ are trained by hiding parts of the audio and predicting them. That is fill-in-the-blank: good embeddings, no generation. A music language model is the "GPT" side.
- **Codec tokens are poor for structure.** Marmoret et al. (2026) tested codec embeddings (DAC, CoDiCodec) for section boundaries and they scored near the bottom. Tokens built to reconstruct sound carry timbre detail, not "this is the chorus again".
- **Section structure is the missing long-range signal.** A section map like the one being tuned here is the kind of plan a generator can be conditioned on to stay coherent over a full song. See [README.md](README.md) for a paper that does exactly this.
