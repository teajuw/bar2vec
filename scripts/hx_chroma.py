"""Per-bar note-content features (chroma): how strongly each of the 12 pitch classes sounds in a bar.

Usage: uv run scripts/hx_chroma.py [--downbeats fold] [--workers 6]

No pretrained model: chroma is computed from the audio with a short-time Fourier transform. Bars are the
same ones used for the per-bar CLAP embeddings, so the two line up one to one.
Writes data/hx_chromabar_<downbeats>.npz (emb = 12 numbers per bar, L2-normalised).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR, HOP = 22050, 512
_bars: dict = {}


N_FFT = 4096


def chroma_map() -> np.ndarray:
    """(12, bins) matrix that adds each frequency bin's energy to its pitch class (A = 440 Hz, 55 Hz to 4 kHz)."""
    freqs = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    m = np.zeros((12, len(freqs)), dtype=np.float32)
    ok = (freqs >= 55) & (freqs <= 4000)
    pc = np.round(12 * np.log2(freqs[ok] / 440.0)).astype(int) % 12
    m[pc, np.flatnonzero(ok)] = 1.0
    return m


def one(t: dict) -> tuple[int, np.ndarray, np.ndarray]:
    # Plain short-time Fourier transform with torch: librosa's chroma functions crash on this machine.
    import torch

    from embed_clap import decode

    audio = decode(ROOT / t["file"], SR)
    a = int(round(t["offset_s"] * SR))
    audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
    db = np.array(t["_downbeats"])
    db = db[db < len(audio) / SR - 0.25]
    spec = torch.stft(torch.from_numpy(audio.copy()), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs().pow(2).numpy()
    chroma = chroma_map() @ spec  # (12, frames)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    edges = np.concatenate([db, [len(audio) / SR]])
    feats = []
    for s, e in zip(edges[:-1], edges[1:]):
        i, j = int(s * SR / HOP), max(int(e * SR / HOP), int(s * SR / HOP) + 1)
        feats.append(chroma[:, i:j].mean(1))
    f = np.array(feats, dtype=np.float32)
    return t["id"], f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9), db.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downbeats", default="fold", choices=["estimated", "fold"])
    ap.add_argument("--split-flow", default="song-cuts")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    src = "hx_downbeats_est.json" if args.downbeats == "estimated" else "hx_downbeats_fold.json"
    downbeats = json.loads((ROOT / "data" / src).read_text())
    tracks = json.loads((ROOT / "data" / "hx_tracks.json").read_text())
    st = json.loads((ROOT / ".claude" / "hillclimb" / args.split_flow / "_state.json").read_text())
    keep = set(st["train_ids"]) | set(st["test_ids"])
    tracks = [{**t, "_downbeats": downbeats[t["name"]]} for t in tracks if t["name"] in keep]
    out = [one(t) for t in tracks]
    np.savez(ROOT / "data" / f"hx_chromabar_{args.downbeats}.npz",
             emb=np.concatenate([f for _, f, _ in out]),
             track=np.concatenate([np.full(len(f), tid, dtype=np.int32) for tid, f, _ in out]),
             start=np.concatenate([d for _, _, d in out]), window=np.float32(2.0))
    print(f"{len(out)} tracks, {sum(len(f) for _, f, _ in out)} bars -> hx_chromabar_{args.downbeats}.npz")


if __name__ == "__main__":
    main()
