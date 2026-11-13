"""Train songs only: save MATPAC++ per-bar embeddings from every layer, for inner-loop exploration.

Usage: uv run scripts/hx_matpac_layers.py
Writes data/hx_matpaclayers_train.npz: emb (bars, 12 layers, 3840) float16, track, start.
"""
import json

import numpy as np
import torch
from tqdm import tqdm

from embed_clap import decode
from hx_embed_matpac import ROOT, SR, load_model


def main() -> None:
    dev = "mps"
    model = load_model(dev)
    st = json.loads((ROOT / ".claude" / "hillclimb" / "song-cuts" / "_state.json").read_text())
    train = set(st["train_ids"])
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in train]
    downbeats = json.loads((ROOT / "data" / "hx_downbeats_fold.json").read_text())
    embs, tid, start = [], [], []
    for t in tqdm(tracks):
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        db = np.array(downbeats[t["name"]])
        db = db[db < len(audio) / SR - 0.25]
        edges = np.concatenate([db, [len(audio) / SR]])
        with torch.no_grad():
            for s, e in zip(edges[:-1], edges[1:]):
                _, layers = model(torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(dev))
                embs.append(layers[0].float().cpu().numpy().astype(np.float16))
        tid += [t["id"]] * len(db)
        start += [float(x) for x in db]
    np.savez(ROOT / "data" / "hx_matpaclayers_train.npz", emb=np.stack(embs), track=np.array(tid, dtype=np.int32), start=np.array(start, dtype=np.float32))
    print(np.stack(embs).shape)


if __name__ == "__main__":
    main()
