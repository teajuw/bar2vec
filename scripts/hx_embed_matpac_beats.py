"""MATPAC++ embedding of one bar's worth of audio starting at EVERY beat (not only at tracked bar starts).

Usage: uv run scripts/hx_embed_matpac_beats.py
For beat j the audio runs from beat j to beat j+k, k = the song's usual beats per bar. This lets the cutter
try other bar phases and place a cut on any beat. Rows whose beat is a tracked bar start match hx_matpacbar_fold.
Writes data/hx_matpacbeat_fold.npz: emb (float16), track, start (beat time), is_down (beat is a tracked bar start), bpb per row.
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
    keep = set(st["train_ids"]) | set(st["test_ids"])
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in keep]
    B = json.loads((ROOT / "data" / "hx_beats_fold.json").read_text())
    embs, tid, start, down, bpbs = [], [], [], [], []
    for t in tqdm(tracks):
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        end = len(audio) / SR
        beats, db = np.array(B[t["name"]]["beats"]), np.array(B[t["name"]]["downbeats"])
        beats = beats[beats < end - 0.25]
        di = np.searchsorted(beats, db - 0.01)
        bpb = int(np.median(np.diff(di))) if len(di) > 2 else 4
        edges = np.concatenate([beats, [end]])
        isd = np.isin(np.arange(len(beats)), di)
        with torch.no_grad():
            for j in range(len(beats)):
                s, e = edges[j], edges[min(len(edges) - 1, j + bpb)]
                emb, _ = model(torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(dev))
                embs.append(emb.float().cpu().numpy()[0])
        tid += [t["id"]] * len(beats); start += beats.tolist(); down += isd.tolist(); bpbs += [bpb] * len(beats)
    emb = np.stack(embs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    np.savez(ROOT / "data" / "hx_matpacbeat_fold.npz", emb=emb.astype(np.float16), track=np.array(tid, dtype=np.int32),
             start=np.array(start, dtype=np.float32), is_down=np.array(down), bpb=np.array(bpbs, dtype=np.int8))
    print(emb.shape)


if __name__ == "__main__":
    main()
