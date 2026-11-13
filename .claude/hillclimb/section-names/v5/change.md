# v5: a verse or bridge with far less singing than the rest of the song is instrumental

v4 never names instrumental (71 on train). Hypothesis: CLAP's text side can tell sung from not sung within a song, even though it cannot tell section roles. Vocal score = best 'singing' prompt minus best 'no vocals' prompt, as a z-score against the song's median section; below -1.0 turns a verse or bridge into instrumental. Train sweep -1.5..-0.5: 55.7 / 56.2 / 56.4 / 56.1 / 54.9. Runner gained extra_prompts (non-label text); v4 reproduced 307/307 after the edit.
