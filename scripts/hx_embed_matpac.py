"""Embed each song one bar at a time with MATPAC++ (music-trained checkpoint).

Usage: uv run scripts/hx_embed_matpac.py [--downbeats fold] [--limit N]

Same bars as the per-bar CLAP embeddings (hx_embed_bars.py), so the two line up one to one.
MATPAC++ takes 16 kHz mono audio of any length; each bar is passed on its own and its frames are
averaged into one 3840-number embedding (the model's default output, as in Marmoret et al. 2026).
The code is loaded from external/matpac without installing it: its installer pins old torch versions.
Writes data/hx_matpacbar_<downbeats>.npz.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from tqdm import tqdm

from embed_clap import decode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "matpac" / "inference_matpac"))
CKPT = "external/matpac_ckpt/matpac_plus_music_6s_2048_enconly.pt"  # relative: the loader branches on "as" in the path
SR = 16000


def load_model(dev: str):
    import functools

    from matpac.model import get_matpac

    assert "as" not in CKPT
    os.chdir(ROOT)
    torch.load = functools.partial(torch.load, weights_only=False, map_location="cpu")  # checkpoint from the authors' release page
    return get_matpac(checkpoint_path=CKPT).to(dev).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downbeats", default="fold", choices=["estimated", "fold"])
    ap.add_argument("--split-flow", default="song-cuts")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="", help="added to the output file name, e.g. _final")
    ap.add_argument("--context", default="1", help="audio given to the encoder for bar i: 1 = the bar alone; "
                    "f2 / f4 = bars i..i+1 / i..i+3; c3 = bars i-1..i+1. Output gets a _<context> suffix.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tracks", default="hx_tracks.json", help="song list under data/")
    args = ap.parse_args()
    model = load_model(args.device)

    tracks = json.loads((ROOT / "data" / args.tracks).read_text())
    st = json.loads((ROOT / ".claude" / "hillclimb" / args.split_flow / "_state.json").read_text())
    keep = set(st["train_ids"]) | set(st["test_ids"])
    tracks = [t for t in tracks if t["name"] in keep][: args.limit or None]
    src = "hx_downbeats_est.json" if args.downbeats == "estimated" else "hx_downbeats_fold.json"
    downbeats = json.loads((ROOT / "data" / src).read_text())

    lo, hi = {"1": (0, 1), "f2": (0, 2), "f4": (0, 4), "c3": (-1, 2)}[args.context]  # bar i uses edges[i+lo] .. edges[i+hi]
    embs, track_ids, start_s, bar_len = [], [], [], []
    for t in tqdm(tracks, desc="bars"):
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        db = np.array(downbeats[t["name"]])
        db = db[db < len(audio) / SR - 0.25]  # same rule as hx_embed_bars.py
        edges = np.concatenate([db, [len(audio) / SR]])
        with torch.no_grad():
            for i in range(len(edges) - 1):
                s, e = edges[max(0, i + lo)], edges[min(len(edges) - 1, i + hi)]
                bar = torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(args.device)
                emb, _ = model(bar)
                embs.append(emb.float().cpu().numpy())
        track_ids += [t["id"]] * len(db)
        start_s += [float(x) for x in db]
        bar_len.append(float(np.median(np.diff(db))))

    emb = np.concatenate(embs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    out = ROOT / "data" / (f"hx_matpacbar_{args.downbeats}{'' if args.context == '1' else '_' + args.context}{args.suffix}.npz" if not args.limit else "hx_matpacbar_test.npz")
    np.savez(out, emb=emb.astype(np.float32), track=np.array(track_ids, dtype=np.int32),
             start=np.array(start_s, dtype=np.float32), window=np.float32(np.median(bar_len)))
    print(f"{len(tracks)} tracks, {emb.shape} bars -> {out.name}; finite: {np.isfinite(emb).all()}")


if __name__ == "__main__":
    main()
