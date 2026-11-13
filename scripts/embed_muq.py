"""Embed every track as overlapping windows with MuQ-MuLan.

Usage: uv run scripts/embed_muq.py [--window 10] [--hop 5]
Uses the track order in data/tracks.json (written by embed_clap.py) so both indexes line up.
Writes data/muq_w<window>_h<hop>.npz.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from tqdm import tqdm

from embed_clap import decode, window_starts

ROOT = Path(__file__).resolve().parent.parent
SR = 24000  # MuQ-MuLan requires 24 kHz


def load_model(device: str):
    from muq import MuQMuLan

    return MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    win, hop = int(args.window * SR), int(args.hop * SR)

    model = load_model(args.device)
    tracks = json.loads((ROOT / "data" / "tracks.json").read_text())

    embs, track_ids, start_s = [], [], []
    for t in tqdm(tracks, desc="tracks"):
        audio = decode(ROOT / t["file"], SR)
        starts = window_starts(len(audio), win, hop)
        chunks = np.stack([np.pad(audio[s : s + win], (0, max(0, win - len(audio[s : s + win])))) for s in starts])
        for b in range(0, len(chunks), args.batch):
            x = torch.from_numpy(chunks[b : b + args.batch]).to(args.device)
            with torch.no_grad():
                e = model.mulan_module.get_audio_latents(x)
            embs.append(e.float().cpu().numpy())
        track_ids += [t["id"]] * len(starts)
        start_s += [s / SR for s in starts]

    emb = np.concatenate(embs).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    tag = f"muq_w{args.window:g}_h{args.hop:g}"
    np.savez(
        ROOT / "data" / f"{tag}.npz",
        emb=emb,
        track=np.array(track_ids, dtype=np.int32),
        start=np.array(start_s, dtype=np.float32),
        window=np.float32(args.window),
    )
    print(f"{len(tracks)} tracks, {len(emb)} windows -> data/{tag}.npz")


if __name__ == "__main__":
    main()
