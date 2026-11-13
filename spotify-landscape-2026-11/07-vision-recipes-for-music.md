# Vision recipes and their music equivalents

Saved 2026-11-11. Transformers treat text, images and audio the same way once the input is cut into tokens; audio is first drawn as a spectrogram (time by pitch) and cut into patches, so most recipes from computer vision carry over directly. "What we have" below means the per-window embeddings in this project (then named `moment-search`) (ten-second windows, five-second hop).

| Vision | Music, using what we have |
|---|---|
| Zero-shot classification with CLIP: score an image against text labels | Score a window against text tags. This is the labelling step for the song breakdown |
| Linear probe: freeze the model, train a small classifier on its vectors | Train a small classifier on window vectors for "intro / verse / drop". Needs labelled songs |
| Segmentation: a label for every patch | A label for every window. This is the song breakdown |
| Object detection: find where the thing is | Moment search: find when the thing is |
| Video model: per-frame vectors fed to a second transformer over time | Per-window vectors fed to a second transformer over the song |

## Notes

- The similarity grid between a song's windows is the first half of attention (a dot product between every pair of tokens) with nothing learned. Real attention only runs inside each ten-second window; nothing in the current pipeline attends across a whole song. The last row of the table is the learned version of that.
- Labelled section data for the linear probe and the second transformer exists publicly (SALAMI, Harmonix). Coverage of electronic music in those sets is unchecked.
- Where the analogy breaks: the two spectrogram axes mean different things (shifting along pitch changes the note, unlike shifting an object in a photo), and a full song is far longer than these models' ten-second input, which is why the windows exist.
