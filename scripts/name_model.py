"""Trained section namer: a classifier over per-section features, fitted on the tuning songs' labels.

Usage: uv run scripts/name_model.py [--extra]   # cross-validate on train songs, fit on all of them, save

--extra adds the training-only songs of hx_tracks_extra.json (partly aligned Harmonix songs; a section
keeps its label only when at least 75% of its bars lie in audio that lines up with the annotated
recording) and writes data/name_model_extra.joblib.

This is the supervised row of the section-names climb, reported beside the training-free namer
(names.py), not in place of it. Features describe a section's place and role in its own song:
position and length, how often its group repeats, what the rule-based namer says about it and its
neighbours, how sung and how loud it is relative to the song, and a few principal components of its
MATPAC++ and CLAP vectors with the song mean removed. Only train songs are read here; train songs
that have a repeat or remix of themselves in the held-out or final sets are left out of the fit.
Writes data/name_model.joblib.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import names as N

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "name_model.joblib"
N_PC = 12
STRUCT = dict(model="matpacbar_fold", position=True, repeat_tau=0.4, bridge=True, prepost=True, short_s=20, vocal_below=-1.0)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def raw_features(bounds, bar_starts, emb_by_model, text) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hand features per section, centred MATPAC++ section vectors, centred CLAP section vectors)."""
    n = len(bounds) - 1
    dur, total = np.diff(bounds), bounds[-1] - bounds[0]
    mat = N.section_vectors(bounds, bar_starts, emb_by_model["matpacbar_fold"])
    clap = N.section_vectors(bounds, bar_starts, emb_by_model["clapbar_fold"])
    d = (clap @ text["vocals"].T).max(1) - (clap @ text["no_vocals"].T).max(1)
    vocal = (d - np.median(d)) / (d.std() + 1e-9)
    level = emb_by_model["loudbar_fold"][:, 0]
    inside = [(bar_starts >= a - 0.05) & (bar_starts < b - 0.05) for a, b in zip(bounds[:-1], bounds[1:])]
    loud = _z(np.array([level[m].mean() if m.any() else level.mean() for m in inside]))
    bars = np.array([m.sum() for m in inside], dtype=float)
    mc = mat - mat.mean(0)
    mc /= np.linalg.norm(mc, axis=1, keepdims=True) + 1e-9
    cc = clap - clap.mean(0)
    cc /= np.linalg.norm(cc, axis=1, keepdims=True) + 1e-9
    sim = mc @ mc.T
    np.fill_diagonal(sim, np.nan)
    g = N.repeat_groups(mat, STRUCT["repeat_tau"])
    count = np.bincount(g)
    rule = N.by_structure(bounds, mat, STRUCT, vocal)
    onehot = np.array([[float(r == l) for l in N.LABELS] for r in rule])
    top = count.max()
    pos = np.arange(n)
    own = [np.flatnonzero(g == g[i]) for i in range(n)]
    base = np.column_stack([
        pos / max(n - 1, 1), pos, n - 1 - pos, bounds[:-1] / total, bounds[1:] / total, np.full(n, n),
        dur, dur / np.median(dur), bars, bars / max(np.median(bars), 1), np.full(n, total),
        count[g], count[g] / n, count[g] == top, count[g] == 1, (count == top).sum() * np.ones(n),
        [np.mean(dur[m]) for m in own], [(m < i).sum() for i, m in enumerate(own)], [m[0] / max(n - 1, 1) for m in own],
        np.nanmax(sim, 1) if n > 1 else np.zeros(n), np.nanmean(sim, 1) if n > 1 else np.zeros(n),
        np.r_[0.0, np.diag(mc @ mc.T, 1)], np.r_[np.diag(mc @ mc.T, 1), 0.0],
        vocal, loud, _z(dur), [vocal[m].mean() for m in own], [loud[m].mean() for m in own],
        np.r_[False, g[1:] == g[:-1]], np.r_[g[:-1] == g[1:], False],
    ]).astype(float)
    local = np.column_stack([base[:, 6:8], base[:, 11:15], base[:, 23:25], onehot])  # what a neighbour contributes
    pad = np.zeros((1, local.shape[1]))
    prev, nxt = np.vstack([pad, local[:-1]]), np.vstack([local[1:], pad])
    prev2, nxt2 = np.vstack([pad, pad, local[:-2]])[:n], np.vstack([local[2:], pad, pad])[:n]
    return np.hstack([base, onehot, prev, nxt, prev2, nxt2]), mc, cc


def design(hand, mc, cc, model) -> np.ndarray:
    return np.hstack([hand, (mc - model["mat_mean"]) @ model["mat_pcs"].T, (cc - model["clap_mean"]) @ model["clap_pcs"].T])


def predict(model, bounds, bar_starts, emb_by_model, text) -> list[str]:
    hand, mc, cc = raw_features(bounds, bar_starts, emb_by_model, text)
    return [N.LABELS[i] for i in model["clf"].predict(design(hand, mc, cc, model))]


def load(which=True):
    """which: True for the model fitted on the tuning songs, "extra" for the one that also saw the extra songs."""
    import joblib

    return joblib.load(MODEL_PATH.with_name("name_model_extra.joblib") if which == "extra" else MODEL_PATH)


def _fit(feats: list, labels: list):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def pcs(x):
        mean = x.mean(0)
        return mean, np.linalg.svd(x - mean, full_matrices=False)[2][:N_PC]
    model = {}
    model["mat_mean"], model["mat_pcs"] = pcs(np.vstack([f[1] for f in feats]))
    model["clap_mean"], model["clap_pcs"] = pcs(np.vstack([f[2] for f in feats]))
    x = np.vstack([design(*f, model) for f in feats])
    y = np.concatenate(labels)
    keep = y >= 0
    model["clf"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
                                                  class_weight="balanced", random_state=0).fit(x[keep], y[keep])
    return model


def extra_sections(text, key):
    """(features, labels, song keys) for the training-only extra songs."""
    import nm_run as R

    chunk_s = 215 * 1024 / 22050  # one alignment chunk in hx_align.py
    z = {m: np.load(ROOT / "data" / f"hx_{m}_extra.npz") for m in ("clapbar_fold", "matpacbar_fold", "loudbar_fold")}
    idx = {l: i for i, l in enumerate(N.LABELS)}
    feats, labels, keys = [], [], []
    for t in json.loads((ROOT / "data" / "hx_tracks_extra.json").read_text()):
        own = z["matpacbar_fold"]["track"] == t["id"]
        starts = z["matpacbar_fold"]["start"][own]
        if own.sum() < 8:
            continue
        bounds, raw = R.reference(t["name"])
        run = t["locked_run"]
        ok = np.array([c < len(run) and run[c] for c in (starts / chunk_s).astype(int)])
        sec = np.clip(np.searchsorted(bounds, starts + 0.05, side="right") - 1, 0, len(raw) - 1)
        y = np.array([idx.get(R.canon(x), -1) for x in raw])
        labels.append(np.array([y[j] if (sec == j).any() and ok[sec == j].mean() >= 0.75 else -1 for j in range(len(raw))]))
        feats.append(raw_features(bounds, starts, {m: v["emb"][own] for m, v in z.items()}, text))
        keys.append(key(t))
    return feats, labels, keys


def main() -> None:
    import argparse

    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true")
    args = ap.parse_args()

    import nm_report as P
    import nm_run as R

    st = json.loads((R.FLOW / "_state.json").read_text())
    split = {n: "train" for n in st["train_ids"]} | {n: "test" for n in st["test_ids"]} | {n: "final" for n in st["final_test_ids"]}
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in split]
    key = lambda t: (re.sub(r"[^a-z0-9]", "", t["title"].lower().split("(")[0]),
                     re.sub(r"[^a-z0-9]", "", t["artist"].lower().split("feat")[0].split(",")[0])[:8])
    where = defaultdict(set)
    for t in tracks:
        where[key(t)].add(split[t["name"]])
    train = [t for t in tracks if split[t["name"]] == "train"]
    fit_on = [t for t in train if where[key(t)] == {"train"}]
    print(f"train songs {len(train)}; fitted on {len(fit_on)} (left out {len(train) - len(fit_on)} with a repeat in held-out or final)")
    cfg = json.loads((ROOT / "names.json").read_text())
    text = R.text_vectors(cfg["encoder"], {**cfg["prompts"], **cfg["extra_prompts"]})
    z = {m: np.load(ROOT / "data" / f"hx_{m}.npz") for m in ("clapbar_fold", "matpacbar_fold", "loudbar_fold")}
    idx = {l: i for i, l in enumerate(N.LABELS)}
    feats, labels, groups = [], [], []
    for t in fit_on:
        bounds, raw = R.reference(t["name"])
        own = z["clapbar_fold"]["track"] == t["id"]
        feats.append(raw_features(bounds, z["clapbar_fold"]["start"][own], {m: v["emb"][v["track"] == t["id"]] for m, v in z.items()}, text))
        labels.append(np.array([idx.get(R.canon(x), -1) for x in raw]))
        groups.append(key(t))
    xf, xl, xk = extra_sections(text, key) if args.extra else ([], [], [])
    if args.extra:
        assert not any("test" in where[k] or "final" in where[k] for k in xk)
        print(f"extra training songs {len(xf)}, labelled sections {sum(int((y >= 0).sum()) for y in xl)}")
    # Cross-validation by song (repeats of a song stay in one fold); PCA and classifier refitted per fold.
    order = sorted(set(groups))
    rng = np.random.default_rng(0)
    rng.shuffle(order)
    fold = {k: i % 5 for i, k in enumerate(order)}
    c = np.zeros((len(fit_on), 8, 8))
    rule = np.zeros((len(fit_on), 8, 8))
    for f in range(5):
        tr = [i for i, k in enumerate(groups) if fold[k] != f]
        ex = [j for j, k in enumerate(xk) if fold.get(k, -1) != f]  # an extra repeat of a tested song stays out
        model = _fit([feats[i] for i in tr] + [xf[j] for j in ex], [labels[i] for i in tr] + [xl[j] for j in ex])
        for i in (i for i, k in enumerate(groups) if fold[k] == f):
            pred = model["clf"].predict(design(*feats[i], model))
            onehot = feats[i][0][:, 30:38].argmax(1)  # the rule-based name, for a like-for-like comparison
            for a, p, r in zip(labels[i], pred, onehot):
                if a >= 0:
                    c[i, a, p] += 1
                    rule[i, a, r] += 1
    b = P.boot(c, rule)
    print(f"out-of-fold on the fitted train songs: trained {b['f1']:.1f} ±{b['f1_ci']:.1f} macro-F1, {b['acc']:.1f} right; "
          f"rules {P.scores(rule.sum(0))[0]:.1f}, {P.scores(rule.sum(0))[1]:.1f}; paired delta {b['d_f1']:+.1f} ±{b['d_f1_ci']:.1f}")
    print("  per label (trained, out-of-fold):", " ".join(f"{l[:5]}:{100 * x:.0f}" for l, x in zip(N.LABELS, P.f1_per_label(c.sum(0)))))
    path = MODEL_PATH.with_name("name_model_extra.joblib") if args.extra else MODEL_PATH
    joblib.dump(_fit(feats + xf, labels + xl), path)
    print(f"saved {path.name}")


if __name__ == "__main__":
    main()
