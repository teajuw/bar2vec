"""Where to cut a song into sections, given its window embeddings. This file and cuts.json are
the surface being tuned by the hill climb; scripts/hx_run.py scores them.
"""
import numpy as np

from breakdown import centred, pick_peaks


def novelty(c: np.ndarray, side: int, guard: int = 0, taper: bool = False) -> np.ndarray:
    """Change score at each window i: how unlike the `side` windows from i onward are to the `side`
    windows that end `guard` windows before i. guard > 0 keeps the two sides from sharing audio.
    taper weights windows near i more than distant ones (a Gaussian, as in Foote's kernel)."""
    s = c @ c.T
    n = len(s)
    nov = np.zeros(n)
    for i in range(2 + guard, n - 1):
        a, b = slice(max(0, i - guard - side), i - guard), slice(i, min(n, i + side))
        if not taper:
            nov[i] = (s[a, a].mean() + s[b, b].mean()) / 2 - s[a, b].mean()
            continue
        wa = np.exp(-0.5 * ((np.arange(a.start, a.stop) - (i - guard)) / (side / 2)) ** 2)
        wb = np.exp(-0.5 * ((np.arange(b.start, b.stop) - i) / (side / 2)) ** 2)
        wa, wb = wa / wa.sum(), wb / wb.sum()
        nov[i] = (wa @ s[a, a] @ wa + wb @ s[b, b] @ wb) / 2 - wa @ s[a, b] @ wb
    return nov


def _vertex(nov: np.ndarray, i: int) -> float:
    """Offset of the true peak from index i, in hops (-0.5..0.5), from a parabola through i-1, i, i+1."""
    a, b, c = nov[i - 1], nov[i], nov[i + 1]
    den = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0


def geometry(cfg: dict, starts: np.ndarray) -> tuple[float, float]:
    """(window length, spacing) in seconds. Bar-aligned windows do not overlap: both are one bar."""
    if cfg.get("bars"):
        bar = float(np.median(np.diff(starts))) if len(starts) > 1 else 2.0
        return bar, bar
    return cfg["window"], cfg["hop"]


def _cbm_cuts(emb_by_model: dict[str, np.ndarray], starts: np.ndarray, cfg: dict):
    """Correlation Block-Matching (Marmoret et al. 2023, package as_seg): choose the segmentation whose
    segments are the most self-similar blocks along the diagonal of the bar-by-bar similarity matrix.
    It rewards stretches that hold together, so it can place a boundary where nothing changes abruptly."""
    from as_seg import CBM_algorithm as cbm
    from as_seg import autosimilarity_computation as asc

    e = next(iter(emb_by_model.values()))
    e = centred(e) if cfg.get("centre", False) else e
    sim = asc.switch_autosimilarity(e, cfg.get("cbm_similarity", "cosine"))
    segments = cbm.compute_cbm(sim, min_size=1, max_size=cfg.get("cbm_max_bars", 32),
                               penalty_weight=cfg.get("cbm_penalty", 0), penalty_func="modulo8",
                               bands_number=cfg.get("cbm_bands"))[0]
    # cbm_shift_bars: when each bar's embedding also covers the following bars, a block starts early by part of that span.
    shift = cfg.get("cbm_shift_bars", 0)
    idx = sorted({int(a) + shift for a, _ in segments if 0 < int(a) + shift < len(starts)})
    nov = np.zeros(len(starts))
    if cfg.get("cbm_support") is not None:
        # Keep a block-matching cut only where the change score also sees something: above the given
        # quantile of the song's positive change scores (0.5 = the median).
        hop = float(np.median(np.diff(starts))) if len(starts) > 1 else 2.0
        # cbm_support_model: name of the model whose change score does the veto (default: the block-matching one).
        veto = emb_by_model[cfg["cbm_support_model"]] if cfg.get("cbm_support_model") else next(iter(emb_by_model.values()))
        nov = novelty(centred(veto), max(1, round(cfg["side_s"] / hop)))
        pos = nov[nov > 0]
        floor = np.quantile(pos, cfg["cbm_support"]) if len(pos) else 0.0
        if cfg.get("cbm_merge_model"):
            # Learned keep/drop on the raw proposals (scripts/merge_model.py, trained on the train songs' annotations),
            # in place of the median veto. Needs CLAP bar embeddings as the model named "clapbar_fold".
            import merge_model

            global _MERGE
            _MERGE = globals().get("_MERGE") or merge_model.load()
            idx = list(merge_model.filter_cuts(_MERGE, next(iter(emb_by_model.values())), starts, idx, nov,
                                               extra={"clap": emb_by_model["clapbar_fold"]}))
        else:
            idx = [i for i in idx if nov[i] > floor]
    return np.array([starts[i] for i in idx], dtype=float), nov


def find_cuts(emb_by_model: dict[str, np.ndarray], starts: np.ndarray, window: float, hop: float, cfg: dict):
    """Return (cut times in seconds, change score per window)."""
    if cfg.get("method") == "cbm":
        return _cbm_cuts(emb_by_model, starts, cfg)
    side = max(1, round(cfg["side_s"] / hop))
    gap = max(1, round(cfg["min_gap_s"] / hop))
    guard = round(cfg.get("guard_s", 0) / hop)
    prep = centred if cfg.get("centre", True) else (lambda e: e / np.linalg.norm(e, axis=1, keepdims=True))
    sides = [max(1, round(x / hop)) for x in cfg["sides_s"]] if cfg.get("sides_s") else [side]
    curves = []
    for e in emb_by_model.values():
        per_scale = [novelty(prep(e), sd, guard, cfg.get("taper", False)) for sd in sides]
        curves.append(np.mean([c / (c.std() + 1e-9) for c in per_scale], axis=0) if len(per_scale) > 1 else per_scale[0])
    if len(curves) > 1:  # different features give change scores on different scales: put each on a unit scale, then weight
        w = cfg.get("weights") or [1.0] * len(curves)
        curves = [wi * c / (c.std() + 1e-9) for wi, c in zip(w, curves)]
    nov = np.mean(curves, axis=0)
    peaks = pick_peaks(nov, gap, cfg["k"])
    # "auto": the change sits in the middle of the audio shared by the last window before and the first after.
    offset = max(0.0, (window - hop * (guard + 1)) / 2) if cfg["offset_s"] == "auto" else cfg["offset_s"]
    refine = cfg.get("refine", True)  # off for bar-aligned windows, where a cut must sit on a bar start
    return np.array([starts[i] + (hop * _vertex(nov, i) if refine else 0.0) for i in peaks], dtype=float) + offset, nov
