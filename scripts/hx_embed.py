"""Embed the Harmonix sample as overlapping windows with CLAP and MuQ-MuLan.

Usage: uv run scripts/hx_embed.py [--window 10] [--hop 5] [--models clap muq]
Reads data/hx_tracks.json, writes data/hx_<model>_w<window>_h<hop>.npz.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from tqdm import tqdm

from embed_clap import CKPT, decode, window_starts

ROOT = Path(__file__).resolve().parent.parent
RATE = {"clap": 48000, "muq": 24000}


def embedder(kind: str, dev: str):
    if kind == "clap":
        import laion_clap

        m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=dev)
        with contextlib.redirect_stdout(io.StringIO()):
            m.load_ckpt(str(CKPT))
        return lambda x: np.asarray(m.get_audio_embedding_from_data(x=x, use_tensor=False), dtype=np.float32)
    from embed_muq import load_model

    m = load_model(dev)

    def run(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return m.mulan_module.get_audio_latents(torch.from_numpy(x).to(dev)).float().cpu().numpy()

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--models", nargs="+", default=["clap", "muq"])
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tracks = json.loads((ROOT / "data" / "hx_tracks.json").read_text())

    for kind in args.models:
        sr = RATE[kind]
        win, hop = int(args.window * sr), int(args.hop * sr)
        run = embedder(kind, dev)
        embs, track_ids, start_s = [], [], []
        for t in tqdm(tracks, desc=kind):
            audio = decode(ROOT / t["file"], sr)
            if "offset_s" in t:  # keep only the span that the annotations cover
                a = int(round(t["offset_s"] * sr))
                audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * sr)]
            starts = window_starts(len(audio), win, hop)
            chunks = np.stack([np.pad(audio[s : s + win], (0, max(0, win - len(audio[s : s + win])))) for s in starts])
            for b in range(0, len(chunks), args.batch):
                embs.append(run(chunks[b : b + args.batch]))
            track_ids += [t["id"]] * len(starts)
            start_s += [s / sr for s in starts]
        emb = np.concatenate(embs).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        tag = f"hx_{kind}_w{args.window:g}_h{args.hop:g}"
        np.savez(ROOT / "data" / f"{tag}.npz", emb=emb, track=np.array(track_ids, dtype=np.int32),
                 start=np.array(start_s, dtype=np.float32), window=np.float32(args.window))
        print(f"{len(tracks)} tracks, {len(emb)} windows -> data/{tag}.npz")


if __name__ == "__main__":
    main()
