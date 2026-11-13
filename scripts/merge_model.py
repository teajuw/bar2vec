"""Learned keep/drop step for block-matching cut proposals (candidate for the song-cut hill climb).

Each proposed cut is described by how the segment before it relates to the segment after it
(lengths in bars, mean and bar-aligned similarity, whether the material recurs elsewhere, position,
change score), a classifier trained on the train songs' annotations scores it, and cuts are dropped
greedily (lowest score first, neighbours re-described after each drop) until every survivor scores
above the threshold.

    model = load()
    kept_idx = filter_cuts(model, emb, starts, proposed_idx, nov, extra={"clap": clap_emb})

Pure numpy at inference. Trained by the script in the scratchpad (merge/train.py); the model file is
data/merge_model.npz.
"""
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "merge_model.npz"
LAGS = (4, 8, 16)
NPC = 8


def _centred(e):
    c = e - e.mean(0)
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)


def _novelty(s, side):
    n = len(s)
    nov = np.zeros(n)
    for i in range(2, n - 1):
        a, b = slice(max(0, i - side), i), slice(i, min(n, i + side))
        nov[i] = (s[a, a].mean() + s[b, b].mean()) / 2 - s[a, b].mean()
    return nov


def _diag4(s):
    """d[i, j] = mean similarity of the 4 bars from i with the 4 bars from j, bar by bar."""
    n = len(s)
    acc, cnt = np.zeros((n, n)), np.zeros((n, n))
    for k in range(4):
        acc[: n - k, : n - k] += s[k:, k:]
        cnt[: n - k, : n - k] += 1
    return acc / cnt


class Song:
    """Per-song tables that every feature reads."""

    def __init__(self, emb, starts, nov, clap=None):
        self.n = n = len(emb)
        self.starts = np.asarray(starts, float)
        self.bar = float(np.median(np.diff(self.starts))) if n > 1 else 2.0
        self.nov = np.asarray(nov, float)
        self.raw = np.asarray(emb, np.float64)
        self.views = {"m": self._view(self.raw)}
        if clap is not None:
            self.views["c"] = self._view(np.asarray(clap, np.float64))
        self.memo = {}

    @staticmethod
    def _view(e):
        c = _centred(e)
        s = c @ c.T
        d4 = _diag4(s)
        n = len(s)
        off = d4[np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 8]
        return dict(c=c, s=s, d4=d4, cum=np.vstack([np.zeros(c.shape[1]), np.cumsum(c, 0)]),
                    ii=np.pad(np.cumsum(np.cumsum(s, 0), 1), ((1, 0), (1, 0))),
                    q90=float(np.quantile(off, 0.9)) if len(off) else 1.0,
                    nov4=_novelty(s, 4), nov8=_novelty(s, 8))


def _block(v, a, b, c, d):
    """Mean of s[a:b, c:d]."""
    ii = v["ii"]
    return (ii[b, d] - ii[a, d] - ii[b, c] + ii[a, c]) / max(1, (b - a) * (d - c))


def _static_view(sg, v, i, tag):
    n, d4 = sg.n, v["d4"]
    f = {}
    for lag in LAGS:
        f[f"{tag}_lagb{lag}"] = d4[i, i - lag] if i - lag >= 0 else np.nan
        f[f"{tag}_lagf{lag}"] = d4[i, i + lag] if i + lag < n else np.nan
    j = np.arange(n)
    past, fut = j <= i - 8, (j >= i + 8) & (j < n - 3)
    far = past | fut
    row = d4[i]
    f[f"{tag}_rec_past"] = row[past].max() if past.any() else np.nan
    f[f"{tag}_rec_fut"] = row[fut].max() if fut.any() else np.nan
    f[f"{tag}_rec_frac"] = float((row[far] > v["q90"]).mean()) if far.any() else np.nan
    # Does what starts here recur after a different lead-in (a section start), and does the lead-in
    # recur with a different continuation (a section end)?
    ok = far & (j >= 4)
    if i >= 4 and ok.any():
        before = np.full(n, np.nan)
        before[4:] = d4[i - 4, : n - 4]
        diff = row[ok] - before[ok]
        f[f"{tag}_newstart"] = diff.max()
        f[f"{tag}_newend"] = (-diff).max()
        top = np.argsort(row[ok])[-3:]
        f[f"{tag}_lead_at_best"] = before[ok][top].mean()
    else:
        f[f"{tag}_newstart"] = f[f"{tag}_newend"] = f[f"{tag}_lead_at_best"] = np.nan
    for nm in ("nov4", "nov8"):
        f[f"{tag}_{nm}"] = v[nm][i]
        f[f"{tag}_{nm}_rank"] = float((v[nm] < v[nm][i]).mean())
    return f


def static_features(sg, i, prop):
    """Features of cut i that do not depend on which other cuts are kept."""
    key = ("s", i)
    if key in sg.memo:
        return sg.memo[key]
    nov, n = sg.nov, sg.n
    pos = nov[nov > 0]
    floor = float(np.quantile(pos, 0.5)) if len(pos) else 0.0
    f = {
        "nov": nov[i], "nov_over_floor": nov[i] / (floor + 1e-9), "nov_rank": float((nov < nov[i]).mean()),
        "nov_rank_prop": float((nov[prop] < nov[i]).mean()) if len(prop) else 0.5,
        "nov_peak": float(nov[i] >= nov[max(0, i - 1)] and nov[i] >= nov[min(n - 1, i + 1)]),
        "pos": i / n, "t_start": sg.starts[i], "t_end": sg.starts[-1] + sg.bar - sg.starts[i],
        "bar_s": sg.bar, "n_bars": n, "prop_density": len(prop) / n,
    }
    for tag, v in sg.views.items():
        f.update(_static_view(sg, v, i, tag))
    sg.memo[key] = f
    return f


def _ctx_view(v, a, i, b, tag):
    L, R = i - a, b - i
    cum = v["cum"]
    ml, mr = (cum[i] - cum[a]) / L, (cum[b] - cum[i]) / R
    cross, wl, wr = _block(v, a, i, i, b), _block(v, a, i, a, i), _block(v, i, b, i, b)
    m = min(L, R)
    k = np.arange(m)
    s = v["s"]
    n = len(s)
    out_l = (_block(v, a, i, 0, n) * n - _block(v, a, i, a, b) * (b - a)) / max(1, n - (b - a))
    out_r = (_block(v, i, b, 0, n) * n - _block(v, i, b, a, b) * (b - a)) / max(1, n - (b - a))
    return {
        f"{tag}_mean_cos": float(ml @ mr / max(1e-9, np.linalg.norm(ml) * np.linalg.norm(mr))),
        f"{tag}_cross": cross, f"{tag}_within_l": wl, f"{tag}_within_r": wr, f"{tag}_seg_nov": (wl + wr) / 2 - cross,
        f"{tag}_align_start": float(s[a + k, i + k].mean()), f"{tag}_align_end": float(s[i - 1 - k, b - 1 - k].mean()),
        f"{tag}_align_gain": float(s[a + k, i + k].mean()) - cross,
        f"{tag}_out_l": out_l, f"{tag}_out_r": out_r,
    }


def context_features(sg, a, i, b):
    """Features of cut i given its neighbouring kept cuts a < i < b (0 and n at the song's ends)."""
    key = (a, i, b)
    if key in sg.memo:
        return sg.memo[key]
    L, R, n, nov = i - a, b - i, sg.n, sg.nov
    f = {"log_l": np.log2(L), "log_r": np.log2(R), "log_min": np.log2(min(L, R)), "log_sum": np.log2(L + R),
         "sum_s": (L + R) * sg.bar, "min_s": min(L, R) * sg.bar, "ratio": np.log2(R / L), "same_len": float(L == R),
         "first": float(a == 0), "last": float(b == n),
         "nov_a": nov[a] if a > 0 else 0.0, "nov_b": nov[b] if b < n else 0.0,
         "nov_vs_nb": nov[i] - max(nov[a] if a > 0 else 0.0, nov[b] if b < n else 0.0)}
    for nm, x in (("l", L), ("r", R), ("sum", L + R)):
        f[f"{nm}_mod4"] = float(x % 4 == 0)
        f[f"{nm}_mod8"] = float(x % 8 == 0)
        f[f"{nm}_mod16"] = float(x % 16 == 0)
    for nm, x in (("l", L), ("r", R)):
        for q in (4, 8, 16):
            f[f"{nm}_is{q}"] = float(x == q)
    for tag, v in sg.views.items():
        f.update(_ctx_view(v, a, i, b, tag))
    sg.memo[key] = f
    return f


def pca_features(sg, i, pcs, mean):
    key = ("p", i)
    if key in sg.memo:
        return sg.memo[key]
    e = sg.raw
    after, before = e[i:i + 8].mean(0), e[max(0, i - 8):i].mean(0)
    d, lvl = (after - before) @ pcs.T, (after - mean) @ pcs.T
    f = {f"p_diff{k}": d[k] for k in range(len(pcs))} | {f"p_lvl{k}": lvl[k] for k in range(len(pcs))}
    sg.memo[key] = f
    return f


def describe(sg, kept, prop, pcs=None, mean=None):
    """One dict of features per kept cut, given the whole kept set."""
    kept = [int(x) for x in kept]
    edges = [0] + kept + [sg.n]
    rows = []
    for k, i in enumerate(kept):
        f = dict(static_features(sg, i, prop))
        f.update(context_features(sg, edges[k], i, edges[k + 2]))
        if pcs is not None:
            f.update(pca_features(sg, i, pcs, mean))
        rows.append(f)
    return rows


def to_matrix(rows, names):
    return np.array([[r[c] for c in names] for r in rows], dtype=float).reshape(len(rows), len(names))


# ---- scoring -----------------------------------------------------------------------------------------------

def predict(model, x):
    """Probability that a cut is a real section boundary."""
    if str(model["kind"]) == "lr":
        z = (np.nan_to_num(x, nan=0.0) - model["mu"]) / model["sd"]
        z = np.where(np.isnan(x), 0.0, z)
        raw = z @ model["coef"] + model["intercept"]
    else:  # gradient-boosted trees exported as flat arrays, one row per tree
        raw = np.full(len(x), float(model["baseline"]))
        feat, thr, left, right, val, leaf, miss = (model[k] for k in ("t_feat", "t_thr", "t_left", "t_right", "t_val", "t_leaf", "t_miss"))
        rows = np.arange(len(x))
        for t in range(len(feat)):
            node = np.zeros(len(x), dtype=int)
            for _ in range(int(model["depth"]) + 1):
                xv = x[rows, feat[t, node]]
                go_left = np.where(np.isnan(xv), miss[t, node], xv <= thr[t, node])
                node = np.where(leaf[t, node], node, np.where(go_left, left[t, node], right[t, node]))
            raw += val[t, node]
    return 1 / (1 + np.exp(-raw))


def decode(model, sg, prop, greedy=True, names=None, pcs=None, mean=None, predict=predict):
    """Return the trajectory [(kept list, lowest score among them)], from all proposals down to none.
    Not greedy: one scoring pass with every proposal present, dropped in score order."""
    names = names if names is not None else [str(c) for c in model["names"]]
    kept = [int(x) for x in prop]
    traj = []
    if not greedy and kept:
        p = predict(model, to_matrix(describe(sg, kept, prop, pcs, mean), names))
        for j in np.argsort(p):
            traj.append((list(kept), float(p[j])))
            kept = [x for x in kept if x != int(prop[j])]
    while greedy and kept:
        p = predict(model, to_matrix(describe(sg, kept, prop, pcs, mean), names))
        j = int(np.argmin(p))
        traj.append((list(kept), float(p[j])))
        kept.pop(j)
    traj.append(([], 1.0))
    return traj


def at_threshold(traj, thr):
    for kept, lowest in traj:
        if lowest >= thr:
            return kept
    return []


def load(path=MODEL_PATH):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def filter_cuts(model, emb, starts, proposed_idx, nov, extra=None):
    """emb: this song's MATPAC bar embeddings; starts: bar start times; proposed_idx: sorted bar indices
    proposed by block matching BEFORE the change-score veto; nov: the per-bar change score from
    _cbm_cuts; extra: {"clap": CLAP bar embeddings for the same bars} when the model uses them."""
    prop = np.array(sorted(int(i) for i in proposed_idx if 0 < int(i) < len(starts)), dtype=int)
    if len(prop) == 0:
        return prop
    uses_clap = bool(model["uses_clap"])
    if uses_clap and not (extra and "clap" in extra):
        raise ValueError("this merge model needs extra={'clap': clap_bar_embeddings}")
    sg = Song(emb, starts, nov, extra["clap"] if uses_clap else None)
    pcs = model["pcs"] if bool(model["uses_pca"]) else None
    traj = decode(model, sg, prop, greedy=bool(model["greedy"]), pcs=pcs, mean=model["pc_mean"] if pcs is not None else None)
    return np.array(at_threshold(traj, float(model["threshold"])), dtype=int)
