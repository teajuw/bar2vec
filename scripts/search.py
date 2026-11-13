"""Search the library by moment, and compare with whole-track baselines.

Usage: uv run scripts/search.py "a heavy drop" [-k 5] [--index clap_w10_h5]

Three ways of ranking, all from the same window embeddings:
  moment     best windows, with adjacent hits in a track merged into one moment
  mean       one vector per track = mean of its windows (a strong whole-track baseline)
  one-crop   one vector per track = a single random window (what vibe does today:
             laion-clap with fusion off embeds one random 10 s crop of the file)
"""
import argparse
import contextlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "music_audioset_epoch_15_esc_90.14.pt"


@dataclass
class Index:
    emb: np.ndarray  # (N, D) L2-normalised window embeddings
    track: np.ndarray  # (N,) track id per window
    start: np.ndarray  # (N,) window start in seconds
    window: float
    tracks: list[dict]

    @classmethod
    def load(cls, name: str = "clap_w10_h5") -> "Index":
        z = np.load(ROOT / "data" / f"{name}.npz")
        tracks = json.loads((ROOT / "data" / "tracks.json").read_text())
        return cls(z["emb"], z["track"], z["start"], float(z["window"]), tracks)

    def mean_vectors(self) -> np.ndarray:
        v = np.stack([self.emb[self.track == t["id"]].mean(0) for t in self.tracks])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def one_crop_vectors(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.stack([self.emb[rng.choice(np.flatnonzero(self.track == t["id"]))] for t in self.tracks])


_models: dict = {}


def embed_text(queries: list[str], index: str = "clap_w10_h5") -> np.ndarray:
    """Text embeddings from whichever model built `index` (clap_* or muq_*), L2-normalised."""
    import torch

    kind = index.split("_")[0]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if kind == "clap":
        if kind not in _models:
            import laion_clap

            m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=dev)
            with contextlib.redirect_stdout(io.StringIO()):
                m.load_ckpt(str(CKPT))
            _models[kind] = m
        padded = list(queries) + [""] if len(queries) == 1 else list(queries)  # library needs >= 2 texts
        e = np.asarray(_models[kind].get_text_embedding(padded, use_tensor=False), dtype=np.float32)[: len(queries)]
    elif kind == "muq":
        if kind not in _models:
            from muq import MuQMuLan

            _models[kind] = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(dev).eval()
        with torch.no_grad():
            e = _models[kind](texts=list(queries)).float().cpu().numpy()
    else:
        raise ValueError(f"unknown index family: {index}")
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def moments(ix: Index, q: np.ndarray, k: int = 5, pool: int = 40, per_track: int = 2) -> list[dict]:
    """Top-k moments: take the `pool` best windows, merge neighbours within a track."""
    scores = ix.emb @ q
    top = np.argsort(-scores)[:pool]
    by_track: dict[int, list[int]] = {}
    for i in top:
        by_track.setdefault(int(ix.track[i]), []).append(int(i))
    out = []
    for tid, idxs in by_track.items():
        idxs.sort(key=lambda i: ix.start[i])
        groups, cur = [], [idxs[0]]
        for i in idxs[1:]:
            if ix.start[i] - ix.start[cur[-1]] <= ix.window:  # overlapping or touching
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        track_moments = []
        for g in groups:
            best = max(g, key=lambda i: scores[i])
            track_moments.append(
                {
                    "track": tid,
                    "start": float(ix.start[g[0]]),
                    "end": float(ix.start[g[-1]] + ix.window),
                    "peak": float(ix.start[best]),
                    "score": float(scores[best]),
                }
            )
        track_moments.sort(key=lambda m: -m["score"])
        out += track_moments[:per_track]
    out.sort(key=lambda m: -m["score"])
    return out[:k]


def rank_tracks(vectors: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.argsort(-(vectors @ q))


def fmt(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--index", default="clap_w10_h5")
    args = ap.parse_args()

    ix = Index.load(args.index)
    q = embed_text([args.query], args.index)[0]
    name = lambda t: f"{ix.tracks[t]['title']} — {ix.tracks[t]['artist']}"

    print(f'\nmoments for "{args.query}"')
    for m in moments(ix, q, args.k):
        print(f"  {m['score']:.3f}  {fmt(m['start'])}–{fmt(m['end'])}  {name(m['track'])}")
    for label, vecs in (("mean of windows", ix.mean_vectors()), ("one random crop", ix.one_crop_vectors())):
        print(f"\nwhole-track baseline ({label})")
        for t in rank_tracks(vecs, q)[: args.k]:
            print(f"  {float(vecs[t] @ q):.3f}  {name(int(t))}")


if __name__ == "__main__":
    main()
