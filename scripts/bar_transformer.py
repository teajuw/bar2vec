"""Bar-level transformer namer: every bar of a song is a token, every bar attends to every other bar.

Usage: uv run scripts/bar_transformer.py cv [--config NAME ...]     # 5-fold cross-validation on train songs

Tokens are per-bar embeddings (MATPAC++ with the song mean removed, reduced to 256 principal components
fitted on the training fold; optionally CLAP the same way, per-bar loudness, and a flag on the first
bar of each given section). The model predicts a label for every bar; a section's name is the label
with the highest mean log-probability over its bars. Scored like the other namers (annotated section
boundaries given, macro-F1 over 8 labels). Only train songs are read; train songs with a repeat or
remix in the held-out or final sets are left out, and repeats stay in one fold.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import names as N
import nm_report as P
import nm_run as R

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "matpac": dict(feats=("matpac",)),
    "matpac+flag": dict(feats=("matpac", "flag")),
    "all": dict(feats=("matpac", "clap", "loud", "flag")),
    "all-small": dict(feats=("matpac", "clap", "loud", "flag"), d=64, layers=2),
    "all-noattn": dict(feats=("matpac", "clap", "loud", "flag"), layers=0),
    "all+rule": dict(feats=("matpac", "clap", "loud", "flag", "rule")),
}
BASE = dict(d=128, layers=3, heads=4, drop=0.3, epochs=60, lr=1e-3, wd=0.05, pcs=256, seeds=2)


def song_key(t):
    return (re.sub(r"[^a-z0-9]", "", t["title"].lower().split("(")[0]),
            re.sub(r"[^a-z0-9]", "", t["artist"].lower().split("feat")[0].split(",")[0])[:8])


CHUNK_S = 215 * 1024 / 22050  # length of one alignment chunk in hx_align.py


def load_songs(extra: bool = False):
    """Per-bar inputs, per-bar label ids (-1 = not scored), section of each bar.

    Default: train songs only. extra=True: the training-only songs of hx_tracks_extra.json, where bars
    outside the chunks that line up with the annotated recording get no label.
    """
    import name_model

    st = json.loads((R.FLOW / "_state.json").read_text())
    split = {n: "train" for n in st["train_ids"]} | {n: "test" for n in st["test_ids"]} | {n: "final" for n in st["final_test_ids"]}
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in split]
    where = defaultdict(set)
    for t in tracks:
        where[song_key(t)].add(split[t["name"]])
    suffix = "_extra" if extra else ""
    z = {m: np.load(ROOT / "data" / f"hx_{m}{suffix}.npz") for m in ("matpacbar_fold", "clapbar_fold", "loudbar_fold")}
    idx = {l: i for i, l in enumerate(N.LABELS)}
    cfg = json.loads((ROOT / "names.json").read_text())
    text = R.text_vectors(cfg["encoder"], {**cfg["prompts"], **cfg["extra_prompts"]})
    if extra:
        tracks = json.loads((ROOT / "data" / "hx_tracks_extra.json").read_text())
    songs = []
    for t in tracks:
        if extra:
            assert "test" not in where[song_key(t)] and "final" not in where[song_key(t)]
        elif split[t["name"]] != "train" or where[song_key(t)] != {"train"}:
            continue
        own = z["matpacbar_fold"]["track"] == t["id"]
        starts = z["matpacbar_fold"]["start"][own]
        bounds, raw = R.reference(t["name"])
        sec = np.clip(np.searchsorted(bounds, starts + 0.05, side="right") - 1, 0, len(raw) - 1)  # section of each bar
        truth = np.array([idx.get(R.canon(x), -1) for x in raw])
        centre = lambda e: e - e.mean(0)
        loud = z["loudbar_fold"]["emb"][own][:, 0]
        y = truth[sec]
        if extra:  # keep labels only where our audio is the annotated recording
            run = t["locked_run"]
            chunk = (starts / CHUNK_S).astype(int)
            y = np.where([c < len(run) and run[c] for c in chunk], y, -1)
        hand = name_model.raw_features(bounds, starts, {m: v["emb"][own] for m, v in z.items()}, text)[0]
        songs.append(dict(name=t["name"], key=song_key(t), sec=sec, truth=truth, y=y, rule=hand[:, 30:38][sec],
                          matpac=centre(z["matpacbar_fold"]["emb"][own]), clap=centre(z["clapbar_fold"]["emb"][own]),
                          loud=((loud - loud.mean()) / (loud.std() + 1e-9))[:, None],
                          flag=np.r_[1.0, (sec[1:] != sec[:-1]).astype(float)][:, None]))
    return songs


class Net(nn.Module):
    def __init__(self, n_in, d, layers, heads, drop):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(drop), nn.Linear(n_in, d))
        self.pos = nn.Linear(4, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True), layers,
                                         enable_nested_tensor=False) if layers else None
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, len(N.LABELS)))

    def forward(self, x, pad):
        n = (~pad).sum(1, keepdim=True).clamp(min=1).float()
        i = torch.arange(x.shape[1], device=x.device)[None].float().expand(x.shape[0], -1)
        rel = i / n
        pos = torch.stack([rel, 1 - rel, torch.sin(i / 8), torch.cos(i / 8)], -1)
        h = self.inp(x) + self.pos(pos)
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=pad)
        return self.out(h)


def batches(items, size, rng=None):
    order = rng.permutation(len(items)) if rng is not None else np.arange(len(items))
    for a in range(0, len(order), size):
        part = [items[j] for j in order[a : a + size]]
        m = max(len(x) for x, _ in part)
        xs = torch.zeros(len(part), m, part[0][0].shape[1])
        ys = torch.full((len(part), m), -1, dtype=torch.long)
        pad = torch.ones(len(part), m, dtype=torch.bool)
        for k, (x, y) in enumerate(part):
            xs[k, : len(x)], ys[k, : len(y)], pad[k, : len(x)] = torch.from_numpy(x), torch.from_numpy(y), False
        yield xs, ys, pad


def build(songs, cfg, proj):
    out = []
    for s in songs:
        parts = []
        for f in cfg["feats"]:
            parts.append((s[f] - proj[f][0]) @ proj[f][1].T / proj[f][2] if f in proj else s[f])
        out.append((np.hstack(parts).astype(np.float32), s["y"]))
    return out


def fit_projection(songs, cfg):
    proj = {}
    for f in ("matpac", "clap"):
        if f in cfg["feats"]:
            x = np.vstack([s[f] for s in songs])
            mean = x.mean(0)
            _, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
            proj[f] = (mean, vt[: cfg["pcs"]], (sv[: cfg["pcs"]] / np.sqrt(len(x)))[None].clip(1e-6) ** 0.5 * (sv[0] / np.sqrt(len(x))) ** 0.5)
    return proj


def train(songs, cfg, seed, dev):
    torch.manual_seed(seed)
    proj = fit_projection(songs, cfg)
    data = build(songs, cfg, proj)
    y = np.concatenate([s["y"] for s in songs])
    w = torch.tensor([len(y[y >= 0]) / (8 * max((y == k).sum(), 1)) for k in range(8)], dtype=torch.float32).clamp(max=5).to(dev)
    net = Net(data[0][0].shape[1], cfg["d"], cfg["layers"], cfg["heads"], cfg["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * int(np.ceil(len(data) / 16))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    for _ in range(cfg["epochs"]):
        net.train()
        for xs, ys, pad in batches(data, 16, rng):
            xs, ys, pad = xs.to(dev), ys.to(dev), pad.to(dev)
            loss = nn.functional.cross_entropy(net(xs, pad).flatten(0, 1), ys.flatten(), weight=w, ignore_index=-1, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    return net.eval(), proj


@torch.no_grad()
def bar_logp(models, songs, cfg, dev):
    """Mean log-probabilities per bar over the given (net, proj) models."""
    out = [0.0] * len(songs)
    for net, proj in models:
        data = build(songs, cfg, proj)
        for k, (xs, _, pad) in enumerate(batches(data, 1)):
            out[k] = out[k] + torch.log_softmax(net(xs.to(dev), pad.to(dev))[0], -1).cpu().numpy() / len(models)
    return out


def section_names(logp, sec, n_sections):
    return np.array([logp[sec == j].mean(0).argmax() if (sec == j).any() else 1 for j in range(n_sections)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv"])
    ap.add_argument("--config", nargs="+", default=list(CONFIGS))
    ap.add_argument("--extra", action="store_true", help="add the training-only extra songs to every training fold")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(8)
    songs = load_songs()
    order = sorted({s["key"] for s in songs})
    np.random.default_rng(0).shuffle(order)  # same folds as the trained-namer cross-validation
    fold = {k: i % 5 for i, k in enumerate(order)}
    extra = load_songs(extra=True) if args.extra else []
    print(f"{len(songs)} train songs, {sum(len(s['y']) for s in songs)} bars; extra training songs {len(extra)}, "
          f"labelled bars {sum(int((s['y'] >= 0).sum()) for s in extra)} of {sum(len(s['y']) for s in extra)}", flush=True)
    for name in args.config:
        cfg = {**BASE, **CONFIGS[name]}
        c = np.zeros((len(songs), 8, 8))
        bar_right = []
        for f in range(5):
            # An extra song that repeats a train song stays out of training when that train song is being tested.
            tr = [s for s in songs if fold[s["key"]] != f] + [s for s in extra if fold.get(s["key"], -1) != f]
            te = [(i, s) for i, s in enumerate(songs) if fold[s["key"]] == f]
            models = [train(tr, cfg, seed, dev) for seed in range(cfg["seeds"])]
            for (i, s), lp in zip(te, bar_logp(models, [s for _, s in te], cfg, dev)):
                for a, p in zip(s["truth"], section_names(lp, s["sec"], len(s["truth"]))):
                    if a >= 0:
                        c[i, a, p] += 1
                bar_right.append((lp.argmax(1) == s["y"])[s["y"] >= 0])
        b = P.boot(c)
        print(f"{name:<14} out-of-fold macro-F1 {b['f1']:.1f} ±{b['f1_ci']:.1f}, sections right {b['acc']:.1f}, bars right {100 * np.concatenate(bar_right).mean():.1f}   "
              + " ".join(f"{l[:4]}:{100 * x:.0f}" for l, x in zip(N.LABELS, P.f1_per_label(c.sum(0)))), flush=True)
        np.save(ROOT / "data" / f"bt_cv_{name}{'_extra' if args.extra else ''}.npy", c)


if __name__ == "__main__":
    main()
