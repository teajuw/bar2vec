"""Fit the trained namer on bar2vec's own cuts and score it end to end, by time.

Usage:
  uv run scripts/nm_cuts_fit.py --cv REGIME [REGIME ...]          # out-of-fold on the tuning songs
  uv run scripts/nm_cuts_fit.py --heldout REGIME --variant NAME   # fit on the tuning songs, score held-out
  uv run scripts/nm_cuts_fit.py --save REGIME                     # fit on the tuning songs, write data/name_model_cuts.joblib

A regime says which sections the classifier is fitted on:
  annotated  the human-drawn sections with their labels (what name_model.py does; the baseline)
  cuts       the cutter's sections, each labelled with the annotated name that covers most of its seconds
  both       the two sets together
  bars       no section features at all: a classifier names every bar from inputs that do not depend on
             where the cuts are (place in the song, how the bar repeats, how sung and loud it is, its
             MATPAC++ and CLAP vectors, each also averaged over the bars around it), and a section takes
             the name its bars vote for
  stack      the annotated regime, with each section's bar votes (and its neighbours') added as inputs
A regime may carry options after a colon, e.g. cuts:pure=0.6 drops cut sections whose main name covers
less than 60% of their labelled seconds.

Scoring is the end-to-end score of nm_e2e.py: the cutter (cuts.json) cuts the song, the classifier names
the cutter's sections, and every second of an annotated, scored section is compared with the name given
to that second. Macro-F1 over the eight labels, pooled over songs, ±95% bootstrap over songs.
--cv reads tuning songs only: five folds by song, classifier refitted per fold, each song named by a
model that never saw it. --heldout is the one look at the held-out songs a round is allowed.
The 180 final songs are never read here.
Writes .claude/hillclimb/section-names-cuts/<variant>/{results.jsonl,summary.json,conf_<split>.npy}.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import name_model as M
import names as N
import nm_run as R
from cuts import find_cuts, geometry
from nm_e2e import IDX, overlap_seconds
from nm_report import boot, f1_per_label

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / ".claude" / "hillclimb" / "section-names-cuts"
MODELS = ("clapbar_fold", "matpacbar_fold", "loudbar_fold")


def song_key(t: dict) -> tuple[str, str]:
    return (re.sub(r"[^a-z0-9]", "", t["title"].lower().split("(")[0]),
            re.sub(r"[^a-z0-9]", "", t["artist"].lower().split("feat")[0].split(",")[0])[:8])


def load_song(t: dict, z: dict, ccfg: dict, text: dict) -> dict:
    """Everything one song contributes: annotated sections, the cutter's sections, features and labels for both."""
    ref_b, raw = R.reference(t["name"])
    truth = [R.canon(x) for x in raw]
    emb = {m: v["emb"][v["track"] == t["id"]] for m, v in z.items()}
    starts = z["clapbar_fold"]["start"][z["clapbar_fold"]["track"] == t["id"]]
    cstarts = z[ccfg["models"][0]]["start"][z[ccfg["models"][0]]["track"] == t["id"]]
    end = float(ref_b[-1])
    cuts, _ = find_cuts({m: emb[m] for m in ccfg["models"]}, cstarts, *geometry(ccfg, cstarts), ccfg)
    cut_b = np.concatenate([[0.0], np.sort(cuts[(cuts > 0) & (cuts < end)]), [end]])
    y_cut, purity = [], []
    for e0, e1 in zip(cut_b[:-1], cut_b[1:]):
        sec = np.zeros(len(N.LABELS))
        for a0, a1, lab in zip(ref_b[:-1], ref_b[1:], truth):
            if lab is not None:
                sec[IDX[lab]] += max(0.0, min(a1, e1) - max(a0, e0))
        y_cut.append(int(sec.argmax()) if sec.sum() > 0 else -1)
        purity.append(float(sec.max() / sec.sum()) if sec.sum() > 0 else 0.0)
    assert len(starts) == len(emb["matpacbar_fold"]) == len(emb["loudbar_fold"])
    sec = np.clip(np.searchsorted(ref_b, starts + 0.05, side="right") - 1, 0, len(truth) - 1)
    y_bar = np.array([IDX[truth[j]] if truth[j] else -1 for j in sec])
    return {"name": t["name"], "key": song_key(t), "ref_b": ref_b, "truth": truth, "cut_b": cut_b,
            "starts": starts, "f_bar": bar_features(starts, end, emb, text), "y_bar": y_bar,
            "f_ann": M.raw_features(ref_b, starts, emb, text), "y_ann": np.array([IDX.get(x, -1) if x else -1 for x in truth]),
            "f_cut": M.raw_features(cut_b, starts, emb, text), "y_cut": np.array(y_cut), "purity": np.array(purity)}


def _smooth(x: np.ndarray, half: int) -> np.ndarray:
    """Mean over the bars within `half` of each bar (fewer at the song's ends)."""
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    i = np.arange(len(x))
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, len(x))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def bar_features(starts: np.ndarray, end: float, emb: dict, text: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hand features per bar, centred MATPAC++ bar vectors, centred CLAP bar vectors). Nothing here uses a section edge."""
    n = len(starts)
    mat, clap = emb["matpacbar_fold"], emb["clapbar_fold"]
    unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    mc, cc = unit(mat - mat.mean(0)), unit(clap - clap.mean(0))
    cu = unit(clap)
    d = (cu @ text["vocals"].T).max(1) - (cu @ text["no_vocals"].T).max(1)
    vocal = (d - np.median(d)) / (d.std() + 1e-9)
    level = emb["loudbar_fold"][:, 0]
    loud = (level - level.mean()) / (level.std() + 1e-9)
    i = np.arange(n)
    sim = mc @ mc.T
    far = np.abs(i[:, None] - i[None, :]) > 2  # repeats, not the bar's own neighbours
    s = np.where(far, sim, -1.0)
    ranked = -np.sort(-s, axis=1)
    alike = (s > 0.5)
    first = np.array([np.flatnonzero(r)[0] if r.any() else k for k, r in enumerate(alike)]) / max(n - 1, 1)
    spread = np.array([np.ptp(np.flatnonzero(r)) if r.any() else 0 for r in alike]) / max(n - 1, 1)
    lag = lambda k: np.array([sim[j, j + k] if 0 <= j + k < n else 0.0 for j in range(n)])
    sm4, sm8 = unit(_smooth(mc, 2)), unit(_smooth(mc, 4))
    ctx = sm8 @ sm8.T
    cs = np.where(np.abs(i[:, None] - i[None, :]) > 8, ctx, -1.0)
    def novelty(h):
        before = unit(np.vstack([mc[max(j - h, 0):j].mean(0) if j else mc[0] for j in range(n)]))
        after = unit(np.vstack([mc[j:j + h].mean(0) for j in range(n)]))
        return 1 - (before * after).sum(1)
    cols = [i / max(n - 1, 1), starts / end, starts, end - starts, np.full(n, n), np.full(n, end), np.r_[np.diff(starts), end - starts[-1]],
            ranked[:, :4].mean(1), ranked[:, :8].mean(1), ranked[:, :16].mean(1), (s > 0.3).mean(1), alike.mean(1), first, spread,
            -np.sort(-cs, axis=1)[:, :8].mean(1), (cs > 0.5).mean(1),
            *[lag(k) for k in (-32, -16, -8, -4, 4, 8, 16, 32)], novelty(4), novelty(8),
            vocal, loud, (sm4 * mc).sum(1)]
    hand = np.column_stack(cols).astype(float)
    vl = np.column_stack([vocal, loud, alike.mean(1), ranked[:, :8].mean(1)])
    hand = np.hstack([hand] + [_smooth(vl, h) for h in (2, 4, 8, 16)] + [np.roll(_smooth(vl, 4), k, 0) for k in (-16, -8, 8, 16)])
    return hand, mc, cc


def bar_design(f, model) -> np.ndarray:
    hand, mc, cc = f
    pm, pc = (mc - model["bmat_mean"]) @ model["bmat_pcs"].T, (cc - model["bclap_mean"]) @ model["bclap_pcs"].T
    return np.hstack([hand, pm, pc, _smooth(pm, 4), _smooth(pc, 4), _smooth(pm, 8)])


def fit_bars(songs: list[dict]) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    def pcs(x):
        mean = x.mean(0)
        return mean, np.linalg.svd(x - mean, full_matrices=False)[2][:M.N_PC]
    model = {}
    model["bmat_mean"], model["bmat_pcs"] = pcs(np.vstack([s["f_bar"][1] for s in songs]))
    model["bclap_mean"], model["bclap_pcs"] = pcs(np.vstack([s["f_bar"][2] for s in songs]))
    x = np.vstack([bar_design(s["f_bar"], model) for s in songs])
    y = np.concatenate([s["y_bar"] for s in songs])
    model["bar_clf"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, l2_regularization=1.0, class_weight="balanced",
                                                      early_stopping=False, random_state=0).fit(x[y >= 0], y[y >= 0])
    return model


def bar_votes(model: dict, s: dict, bounds: np.ndarray) -> np.ndarray:
    """(sections x labels): mean log-probability its bars give each name. A section with no bar start takes the nearest bar."""
    logp = np.full((len(s["starts"]), len(N.LABELS)), -20.0)
    logp[:, model["bar_clf"].classes_] = np.log(model["bar_clf"].predict_proba(bar_design(s["f_bar"], model)) + 1e-9)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = (s["starts"] >= a - 0.05) & (s["starts"] < b - 0.05)
        out.append(logp[m].mean(0) if m.any() else logp[np.abs(s["starts"] - (a + b) / 2).argmin()])
    return np.array(out)


def with_votes(f, votes: np.ndarray):
    """Section features plus the section's bar votes and its neighbours' votes."""
    pad = np.zeros((1, votes.shape[1]))
    return np.hstack([f[0], votes, np.vstack([pad, votes[:-1]]), np.vstack([votes[1:], pad])]), f[1], f[2]


def fit(songs: list[dict], regime: str) -> dict:
    base, _ = parse(regime)
    if base == "bars":
        return fit_bars(songs)
    if base == "stack":
        # The section classifier must see votes from a bar model that never saw the song, as it will at test time.
        order = sorted({s["key"] for s in songs})
        np.random.default_rng(1).shuffle(order)
        inner = {k: i % 4 for i, k in enumerate(order)}
        feats = [None] * len(songs)
        for f in range(4):
            bm = fit_bars([s for s in songs if inner[s["key"]] != f])
            for i, s in enumerate(songs):
                if inner[s["key"]] == f:
                    feats[i] = with_votes(s["f_ann"], bar_votes(bm, s, s["ref_b"]))
        return {**M._fit(feats, [s["y_ann"] for s in songs]), **fit_bars(songs), "stack": True}
    return M._fit(*training_set(songs, regime))


def parse(regime: str) -> tuple[str, dict]:
    base, _, opts = regime.partition(":")
    return base, {k: float(v) for k, v in (o.split("=") for o in opts.split(",") if o)}


def training_set(songs: list[dict], regime: str) -> tuple[list, list]:
    base, opt = parse(regime)
    assert base in ("annotated", "cuts", "both"), base
    feats, labels = [], []
    for s in songs:
        if base in ("annotated", "both"):
            feats.append(s["f_ann"])
            labels.append(s["y_ann"])
        if base in ("cuts", "both"):
            feats.append(s["f_cut"])
            labels.append(np.where(s["purity"] >= opt.get("pure", 0.0), s["y_cut"], -1))
    return feats, labels


def name_cuts(model: dict, s: dict) -> np.ndarray:
    """Seconds of (annotated label, given name) for one song: the model names the cutter's sections."""
    if "clf" not in model:
        pred = [N.LABELS[i] for i in bar_votes(model, s, s["cut_b"]).argmax(1)]
    elif model.get("stack"):
        pred = [N.LABELS[i] for i in model["clf"].predict(M.design(*with_votes(s["f_cut"], bar_votes(model, s, s["cut_b"])), model))]
    else:
        pred = [N.LABELS[i] for i in model["clf"].predict(M.design(*s["f_cut"], model))]
    return overlap_seconds(s["ref_b"], s["truth"], s["cut_b"], pred)


def write(variant: str, split: str, regime: str, songs: list[dict], conf: np.ndarray, note: str) -> dict:
    out = FLOW / variant
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"conf_{split}.npy", conf)
    rows = [{"prompt_id": s["name"], "rep": 0, "tags": [split],
             "grade": {"seconds_right": round(float(np.trace(c) / max(c.sum(), 1e-9)), 4)},
             "meta": {"sections_cut": len(s["cut_b"]) - 1, "sections_annotated": len(s["ref_b"]) - 1}}
            for s, c in zip(songs, conf)]
    path = out / "results.jsonl"
    old = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    old = [r for r in old if r["tags"][0] != split]
    path.write_text("".join(json.dumps(r) + "\n" for r in old + rows))
    b = boot(conf)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"description": note, "target": "code", "regime": regime}
    summary[split] = {"songs": len(songs), "macro_f1_time": round(b["f1"], 1), "macro_f1_ci": round(b["f1_ci"], 1),
                      "seconds_right": round(b["acc"], 1), "seconds_right_ci": round(b["acc_ci"], 1),
                      "f1_per_label": {l: round(100 * float(x), 1) for l, x in zip(N.LABELS, f1_per_label(conf.sum(0)))}}
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", nargs="+", metavar="REGIME")
    ap.add_argument("--heldout", metavar="REGIME")
    ap.add_argument("--save", metavar="REGIME")
    ap.add_argument("--variant", default="")
    ap.add_argument("--versus", default="baseline", help="variant to pair the held-out score against")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    st = json.loads((R.FLOW / "_state.json").read_text())
    split = {n: "train" for n in st["train_ids"]} | {n: "test" for n in st["test_ids"]} | {n: "final" for n in st["final_test_ids"]}
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in split]
    where = defaultdict(set)
    for t in tracks:
        where[song_key(t)].add(split[t["name"]])
    fit_tracks = [t for t in tracks if split[t["name"]] == "train" and where[song_key(t)] == {"train"}]
    ccfg = json.loads((ROOT / "cuts.json").read_text())
    cfg = json.loads((ROOT / "names.json").read_text())
    text = R.text_vectors(cfg["encoder"], {**cfg["prompts"], **cfg["extra_prompts"]})
    z = {m: np.load(ROOT / "data" / f"hx_{m}.npz") for m in set(MODELS) | set(ccfg["models"])}
    train = [load_song(t, z, ccfg, text) for t in fit_tracks]
    print(f"tuning songs fitted on: {len(train)}; sections annotated {sum(len(s['y_ann']) for s in train)}, cut {sum(len(s['y_cut']) for s in train)}; "
          f"cut sections whose main name covers under 60% of them: {100 * np.mean(np.concatenate([s['purity'] for s in train]) < 0.6):.0f}%")

    if args.cv:
        order = sorted({s["key"] for s in train})
        np.random.default_rng(0).shuffle(order)
        fold = {k: i % 5 for i, k in enumerate(order)}
        conf = {}
        for regime in args.cv:
            c = np.zeros((len(train), 8, 8))
            for f in range(5):
                model = fit([s for s in train if fold[s["key"]] != f], regime)
                for i, s in enumerate(train):
                    if fold[s["key"]] == f:
                        c[i] = name_cuts(model, s)
            conf[regime] = c
            b = boot(c, conf[args.cv[0]]) if regime != args.cv[0] else boot(c)
            delta = f"   vs {args.cv[0]}: {b['d_f1']:+.1f} ±{b['d_f1_ci']:.1f} macro-F1, {b['d_acc']:+.1f} ±{b['d_acc_ci']:.1f} seconds" if "d_f1" in b else ""
            print(f"  out-of-fold {regime:<16} macro-F1 {b['f1']:5.1f} ±{b['f1_ci']:.1f}   seconds right {b['acc']:5.1f} ±{b['acc_ci']:.1f}{delta}")
            print("      " + "  ".join(f"{l[:6]} {100 * x:3.0f}" for l, x in zip(N.LABELS, f1_per_label(c.sum(0)))))
            if args.variant and regime == args.cv[-1]:
                write(args.variant, "train", regime, train, c, args.note)

    if args.heldout:
        assert args.variant, "--heldout needs --variant"
        test = [load_song(t, z, ccfg, text) for t in tracks if split[t["name"]] == "test"]
        model = fit(train, args.heldout)
        c = np.stack([name_cuts(model, s) for s in test])
        b = write(args.variant, "test", args.heldout, test, c, args.note)
        print(f"  held-out {args.heldout} ({len(test)} songs): macro-F1 {b['f1']:.1f} ±{b['f1_ci']:.1f}, seconds right {b['acc']:.1f} ±{b['acc_ci']:.1f}")
        ref = FLOW / args.versus / "conf_test.npy"
        if args.variant != args.versus and ref.exists():
            d = boot(c, np.load(ref))
            print(f"  paired vs {args.versus}: macro-F1 {d['d_f1']:+.1f} ±{d['d_f1_ci']:.1f}, seconds right {d['d_acc']:+.1f} ±{d['d_acc_ci']:.1f}")

    if args.save:
        import joblib

        joblib.dump(fit(train, args.save), ROOT / "data" / "name_model_cuts.joblib")
        print("saved data/name_model_cuts.joblib")


if __name__ == "__main__":
    main()
