"""Where to cut a song into sections, given its window embeddings. This file and cuts.json are
the surface being tuned by the hill climb; scripts/hx_run.py scores them.
"""
import numpy as np

from breakdown import centred, novelty, pick_peaks


def find_cuts(emb_by_model: dict[str, np.ndarray], starts: np.ndarray, window: float, hop: float, cfg: dict):
    """Return (cut times in seconds, change score per window)."""
    side = max(1, round(cfg["side_s"] / hop))
    gap = max(1, round(cfg["min_gap_s"] / hop))
    nov = np.mean([novelty(centred(e), side) for e in emb_by_model.values()], axis=0)
    peaks = pick_peaks(nov, gap, cfg["k"])
    return np.array([starts[i] for i in peaks], dtype=float) + cfg["offset_s"], nov
