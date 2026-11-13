# Music language models: papers and what they mean for this project

Collected 2026-11-12. Ten papers from arXiv. The PDFs are kept locally and are not in the repository; the table links to each paper. The summaries below are from the papers' abstracts and, where noted, their method sections. I read the abstracts of all ten and one method section (YuE 3.2.2); I have not read the papers in full.

The plain-language framing is in [00-llm-to-music-model.md](00-llm-to-music-model.md).

## The main finding for us

**The strongest open long-form generator uses a section cutter as a core part of its recipe.** YuE (2025) could not get lyrics and structure to hold over a full song by lengthening the context: text conditioning given at the start faded after about 3,000 audio tokens and failed past 6,000. Its fix was to cut every training song into sections with an automatic segmenter, then interleave a text label, the lyrics and the audio tokens section by section. The segmenter it used is All-In-One, a supervised model trained on Harmonix.

That is the job our cutter does. The paper we benchmark against (Marmoret et al. 2026) lists All-In-One as the supervised system that training-free cutters trail by a few points.

## The papers

| Link | Paper | Kind | One-line summary |
|---|---|---|---|
| [arXiv 2005.00341](https://arxiv.org/abs/2005.00341) | Jukebox (OpenAI) | tokens + transformer | First to generate minutes of music with singing as raw audio: a multiscale VQ-VAE compresses audio to codes, autoregressive transformers model them. |
| [arXiv 2107.03312](https://arxiv.org/abs/2107.03312) | SoundStream (Google) | tokenizer | The codec design later models build on: encoder, residual vector quantizer, decoder, trained end to end. |
| [arXiv 2210.13438](https://arxiv.org/abs/2210.13438) | EnCodec (Meta) | tokenizer | Meta's codec of the same design; the tokenizer inside MusicGen. |
| [arXiv 2306.06546](https://arxiv.org/abs/2306.06546) | Descript Audio Codec | tokenizer | Improved codec of the same family; one of the codecs Marmoret tested for structure. |
| [arXiv 2208.12415](https://arxiv.org/abs/2208.12415) | MuLan (Google) | text-audio embedding | Joint embedding of music and text, the steering signal for MusicLM. Our MuQ-MuLan is a later model of this type. |
| [arXiv 2209.03143](https://arxiv.org/abs/2209.03143) | AudioLM (Google) | tokens + transformer | Uses two kinds of token: "semantic" tokens from a masked audio model for long-term structure, and codec tokens for sound quality. |
| [arXiv 2301.11325](https://arxiv.org/abs/2301.11325) | MusicLM (Google) | tokens + transformer | AudioLM steered by MuLan: text to music at 24 kHz, consistent over several minutes. Released the MusicCaps dataset (5.5k clips with expert captions). |
| [arXiv 2306.05284](https://arxiv.org/abs/2306.05284) | MusicGen (Meta) | tokens + transformer | One transformer over several interleaved code streams, with text or melody conditioning. Open weights and code. |
| [arXiv 2402.04825](https://arxiv.org/abs/2402.04825) | Stable Audio (Stability) | diffusion | Latent diffusion, steered by a CLAP text encoder plus timing inputs so it can produce a chosen length; up to 95 s of stereo at 44.1 kHz. |
| [arXiv 2503.08638](https://arxiv.org/abs/2503.08638) | YuE | tokens + transformer | Open lyrics-to-song model, up to five minutes, built on an LLM architecture. Section-by-section conditioning as described above. |

## What we can learn

1. **Two kinds of token, and we are working with one of them.** AudioLM's split is the clearest statement: tokens from a masked-prediction model carry long-term structure; codec tokens carry sound quality. MATPAC++ and MuQ are masked-prediction models. This is the generation-side version of what our boundary scores show (MATPAC++ strong, codec embeddings weak in Marmoret's tests).
2. **Structure does not emerge from a longer context alone.** Jukebox, MusicLM and MusicGen all name long-range structure as the hard part. YuE reports the specific failure and solves it with explicit section labels.
3. **The steering model matters and is shared with search.** MusicLM uses MuLan, Stable Audio uses a CLAP text encoder. The text-audio models we use for moment search are the same component generators use to follow a prompt.
4. **Timing as an input.** Stable Audio conditions on start time and total length, because training on random crops makes a model produce "arbitrary sections of a song, which may start or end in the middle of a musical phrase". Cutting on section boundaries instead of at random is the obvious alternative, and needs a cutter.

## How it applies to this project

- **A use for the cutter beyond the playlist page.** A training-free cutter that works on any music could prepare training data for generators in genres where a Harmonix-trained supervised segmenter is weak. This is a claim to test, not a result: we have only measured on Harmonix, which is Western pop.
- **Section naming matters more than I had rated it.** YuE needs a label per section ([intro], [verse], [chorus], [bridge], [outro]), not only the cuts. Our plan had section naming as a later item. This is a reason to keep it.
- **A third eval for the brief.** Beyond boundary accuracy, the cutter could be judged by whether section-wise conditioning with our cuts works as well as with All-In-One's. That needs a generator and is out of reach on this machine today; noted as a direction.
- **A check we could run now.** YuE's paper says its learned representations do well on music-understanding benchmarks. Whether a generator's internal features find boundaries better than MATPAC++ is the same experiment as round 8 with a different encoder. YuE is large, so this may not fit the 64 GB Mac mini; unverified.

## Not covered

- Suno and Udio have not published their methods (YuE says the same).
- Riffusion (diffusion on spectrogram images) has no paper; it is mentioned in the framing note only.
- Symbolic (MIDI) music models are a separate line and not collected here.
