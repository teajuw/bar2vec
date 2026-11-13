# Metrics

All are boundary F-scores from mir_eval, 0 to 1 per song, higher is better. A predicted cut is a hit when it lies within the tolerance of an annotated section boundary; F combines hits per prediction and hits per annotation.

- **f3_trim (target):** 3-second tolerance, first and last boundary of each song dropped (they are free hits).
- **f05_trim (guardrail):** half-second tolerance, trimmed.
- **f3, f05:** the same without trimming, for comparison with published numbers.

Cases: Harmonix Set songs whose YouTube audio is within 2 s of the annotated length. Annotations are human-made by Harmonix. One rep per song: the pipeline is deterministic.
