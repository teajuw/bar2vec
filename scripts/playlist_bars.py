"""Track bars in the playlist songs and embed each bar with MATPAC++ (the bar2vec cutter's input).

Usage: uv run scripts/playlist_bars.py

Bars come from the Beat This! tracker's default model. Each bar is passed to MATPAC++ on its own,
as in hx_embed_matpac.py. Writes data/pl_downbeats.json ({track id: {beats, downbeats}}) and
data/pl_matpacbar.npz (emb, track, start).
"""
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from embed_clap import decode
from hx_embed_matpac import SR, load_model

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from beat_this.inference import Audio2Beats

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tracker = Audio2Beats(checkpoint_path="final0", device=dev, dbn=False)
    model = load_model(dev)
    tracks = json.loads((ROOT / "data" / "tracks.json").read_text())
    grid, embs, track_ids, start_s = {}, [], [], []
    for t in tqdm(tracks, desc="bars"):
        audio = decode(ROOT / t["file"], SR)
        beats, downbeats = tracker(audio, SR)
        db = np.asarray(downbeats, dtype=float)
        db = db[db < len(audio) / SR - 0.25]
        grid[t["id"]] = {"beats": [round(float(x), 3) for x in beats], "downbeats": [round(float(x), 3) for x in db]}
        edges = np.concatenate([db, [len(audio) / SR]])
        with torch.no_grad():
            for s, e in zip(edges[:-1], edges[1:]):
                emb, _ = model(torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(dev))
                embs.append(emb.float().cpu().numpy())
        track_ids += [t["id"]] * len(db)
        start_s += [float(x) for x in db]
    emb = np.concatenate(embs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    (ROOT / "data" / "pl_downbeats.json").write_text(json.dumps(grid))
    np.savez(ROOT / "data" / "pl_matpacbar.npz", emb=emb.astype(np.float32), track=np.array(track_ids, dtype=np.int32),
             start=np.array(start_s, dtype=np.float32))
    bars = [len(g["downbeats"]) for g in grid.values()]
    print(f"{len(tracks)} songs, {len(emb)} bars (per song min {min(bars)}, median {int(np.median(bars))}); finite: {np.isfinite(emb).all()}")


if __name__ == "__main__":
    main()
