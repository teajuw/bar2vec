"""Name each section of a song. This file and names.json are the surface tuned by the section-names
hill climb; scripts/nm_run.py scores them. Nothing here is trained and nothing here sees an annotation label.
"""
import numpy as np

LABELS = ["intro", "verse", "prechorus", "chorus", "postchorus", "bridge", "outro", "instrumental"]


def section_vectors(bounds: np.ndarray, bar_starts: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """One unit vector per section: the mean of the bars that start inside it (nearest bar if none do)."""
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        inside = (bar_starts >= a - 0.05) & (bar_starts < b - 0.05)
        v = emb[inside].mean(0) if inside.any() else emb[np.argmin(np.abs(bar_starts - a))]
        out.append(v / max(np.linalg.norm(v), 1e-9))
    return np.stack(out)


def name_sections(bounds: np.ndarray, bar_starts: np.ndarray, emb_by_model: dict[str, np.ndarray],
                  text: dict[str, np.ndarray], cfg: dict) -> tuple[list[str], np.ndarray]:
    """Return (one label per section, score matrix sections x LABELS).

    bounds: section edges in seconds (n+1 values). text: label -> (prompts, dim) unit vectors from the
    text side of cfg["encoder"]. A section gets the label whose best prompt is closest to its audio.
    """
    vecs = section_vectors(bounds, bar_starts, emb_by_model[cfg["encoder"]])
    scores = np.stack([(vecs @ text[label].T).max(1) for label in LABELS], axis=1)
    return [LABELS[i] for i in scores.argmax(1)], scores
