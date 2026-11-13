"""Learned beat placer: move each kept cut to one of the beats j-2..j+2.

A conditional logit (softmax over the five candidate beats, one shared weight vector) on
per-beat scalar features z-scored within the song, plus a song-level second pass that adds
lam * (the song's mean per-offset log-probability over its cuts, minus the training mean), then a stay margin.
Pure numpy at inference.

    model = load()
    new_cut_beats = place(model, song, cut_beats)

song = {"bar_emb", "mel", "beat1" (or None), "start", "is_down", "bpb"} for one song's beats.
"""
from pathlib import Path

import numpy as np

OFFS = np.array([-2, -1, 0, 1, 2])
MODEL = Path(__file__).resolve().parent.parent / "data" / "beat_placer.npz"

GROUPS = {
    "bar": ["m_pair", "m_w2", "m_hom_after", "m_hom_before", "m_step"],
    "mel": ["flux", "dlev", "abs_dlev", "lev_prev", "spec1", "speck", "spec2k", "dlevk", "abs_dlevk", "dbass"],
    "beat1": ["b_step", "b_k", "b_rep", "b_rep_prev"],
}


def _rows_dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _cos(a, b):
    return _rows_dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def _win_mean(x, k):
    """w[i] = mean(x[i:i+k]); nan rows where the window runs past the end."""
    n = len(x)
    c = np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), np.cumsum(x, 0, dtype=np.float64)])
    out = np.full(x.shape, np.nan, np.float64)
    if n >= k:
        out[: n - k + 1] = (c[k:] - c[:-k]) / k
    return out


def _shift(a, d):
    """b[x] = a[x-d], nan where undefined (works for 1-d and 2-d)."""
    out = np.full(a.shape, np.nan, np.float64)
    n = len(a)
    if d == 0:
        out[:] = a
    elif 0 < d < n:
        out[d:] = a[: n - d]
    elif d < 0 and -d < n:
        out[:d] = a[-d:]
    return out


def beat_features(song, groups):
    """(n_beats, F) scalar features, z-scored within the song, nan -> 0. Returns (X, names)."""
    k = int(song["bpb"]) if int(song["bpb"]) > 0 else 4
    n = len(song["start"])
    cols, names = [], []
    if "bar" in groups:
        e = np.asarray(song["bar_emb"], np.float32).astype(np.float64)
        prev, prev2, nxt = _shift(e, k), _shift(e, 2 * k), _shift(e, -k)
        cols += [1 - _rows_dot(prev, e), 1 - _cos(prev + prev2, e + nxt), _rows_dot(e, nxt), _rows_dot(prev, prev2),
                 1 - _rows_dot(_shift(e, 1), e)]
        names += GROUPS["bar"]
    if "mel" in groups:
        m = np.asarray(song["mel"], np.float64)
        L, lev, flux = m[:, :64], m[:, 64], m[:, 65]
        aft, aft2 = _win_mean(L, k), _win_mean(L, 2 * k)
        bef, bef2 = _shift(aft, k), _shift(aft2, 2 * k)
        la = _win_mean(lev[:, None], k)[:, 0]
        dl, dlk = lev - _shift(lev, 1), la - _shift(la, k)
        cols += [flux, dl, np.abs(dl), _shift(lev, 1), np.linalg.norm(L - _shift(L, 1), axis=1),
                 np.linalg.norm(aft - bef, axis=1), np.linalg.norm(aft2 - bef2, axis=1), dlk, np.abs(dlk),
                 (L[:, :8] - _shift(L[:, :8], 1)).mean(1)]
        names += GROUPS["mel"]
    if "beat1" in groups:
        b = np.asarray(song["beat1"], np.float32).astype(np.float64)
        aft = _win_mean(b, k)
        rep = 1 - _cos(_shift(b, k), b)
        cols += [1 - _cos(_shift(b, 1), b), 1 - _cos(_shift(aft, k), aft), rep, _shift(rep, 1)]
        names += GROUPS["beat1"]
    X = np.stack(cols, 1) if cols else np.zeros((n, 0))
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    X = (X - mu) / (sd + 1e-9)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def candidate_features(song, cut_beats, groups, rank=False):
    """(n_cuts, 5, F) features and (n_cuts, 5) validity mask. Offsets get one-hot intercepts."""
    Xb, names = beat_features(song, groups)
    n = len(Xb)
    cut = np.asarray(cut_beats, int).reshape(-1)
    idx = cut[:, None] + OFFS[None, :]
    valid = (idx >= 0) & (idx < n)
    F = Xb[np.clip(idx, 0, n - 1)]
    parts = [F]
    if rank:  # is this candidate the largest of the five on each feature
        Fm = np.where(valid[:, :, None], F, -np.inf)
        parts.append((Fm >= Fm.max(1, keepdims=True)).astype(np.float64))
    onehot = np.zeros((len(cut), 5, 4))
    for c, o in enumerate([0, 1, 3, 4]):
        onehot[:, o, c] = 1
    parts.append(onehot)
    three = float(int(song["bpb"]) == 3)
    parts.append(onehot * three)
    return np.concatenate(parts, 2), valid


def log_probs(w, F, valid):
    s = np.where(valid, F @ w, -1e9)
    s = s - s.max(1, keepdims=True)
    return s - np.log(np.exp(s).sum(1, keepdims=True))


def decide(lp, valid, lam, margin, prior=None):
    """Second pass: add lam * (song-mean log-prob per offset, minus the training-set mean `prior` when
    given, so that only the song's departure from a typical song counts); move only if the best
    offset beats staying by `margin`."""
    if len(lp) == 0:
        return np.zeros(0, int)
    song = np.nanmean(np.where(valid, lp, np.nan), 0) if valid.any(0).all() else np.zeros(5)
    if prior is not None:
        song = song - prior
    s = np.where(valid, lp + lam * song[None, :], -1e9)
    best = s.argmax(1)
    gain = s[np.arange(len(s)), best] - s[:, 2]
    return np.where(gain > margin, OFFS[best], 0)


def load(path=MODEL):
    z = np.load(path, allow_pickle=False)
    return {"w": z["w"], "groups": [str(g) for g in z["groups"]], "rank": bool(z["rank"]),
            "lam": float(z["lam"]), "margin": float(z["margin"]),
            "prior": z["prior"] if "prior" in z.files else None}


def place(model, song, cut_beats):
    cut = np.asarray(cut_beats, int).reshape(-1)
    if len(cut) == 0:
        return cut
    groups = [g for g in model["groups"]]
    if "beat1" in groups and song.get("beat1") is None:
        raise ValueError("model needs single-beat embeddings (song['beat1'])")
    F, valid = candidate_features(song, cut, groups, model["rank"])
    lp = log_probs(model["w"], F, valid)
    return cut + decide(lp, valid, model["lam"], model["margin"], model.get("prior"))
