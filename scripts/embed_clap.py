"""Embed every track as overlapping windows with CLAP (music checkpoint).

Usage: uv run scripts/embed_clap.py [--window 10] [--hop 5]
Reads data/downloads.json + audio/, writes data/clap_w<window>_h<hop>.npz and data/tracks.json.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SR = 48000  # CLAP's expected sample rate
CKPT = ROOT / "models" / "music_audioset_epoch_15_esc_90.14.pt"


def decode(path: Path, sr: int = SR) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr` using ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32)


def window_starts(n_samples: int, win: int, hop: int) -> list[int]:
    """Start offsets of full windows, plus one final window flush with the end."""
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples:
        starts.append(n_samples - win)
    return starts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    win, hop = int(args.window * SR), int(args.hop * SR)

    import laion_clap

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=dev)
    import contextlib, io

    with contextlib.redirect_stdout(io.StringIO()):  # load_ckpt prints every tensor name
        model.load_ckpt(str(CKPT))

    downloads = json.loads((ROOT / "data" / "downloads.json").read_text())
    rows, seen = [], set()
    for r in sorted((r for r in downloads.values() if "file" in r), key=lambda r: r["index"]):
        key = (r["title"], r["artist"])  # the playlist lists one song twice under two URIs
        if key not in seen:
            seen.add(key)
            rows.append(r)

    embs, track_ids, start_s, tracks = [], [], [], []
    for tid, r in enumerate(tqdm(rows, desc="tracks")):
        audio = decode(ROOT / r["file"])
        starts = window_starts(len(audio), win, hop)
        chunks = np.stack([np.pad(audio[s : s + win], (0, max(0, win - len(audio[s : s + win])))) for s in starts])
        for b in range(0, len(chunks), args.batch):
            e = model.get_audio_embedding_from_data(x=chunks[b : b + args.batch], use_tensor=False)
            embs.append(np.asarray(e, dtype=np.float32))
        track_ids += [tid] * len(starts)
        start_s += [s / SR for s in starts]
        tracks.append(
            {
                "id": tid,
                "index": r["index"],
                "title": r["title"],
                "artist": r["artist"],
                "file": r["file"],
                "duration_s": round(len(audio) / SR, 1),
                "n_windows": len(starts),
                "flag": r.get("flag", False),
            }
        )

    emb = np.concatenate(embs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    tag = f"clap_w{args.window:g}_h{args.hop:g}"
    np.savez(
        ROOT / "data" / f"{tag}.npz",
        emb=emb,
        track=np.array(track_ids, dtype=np.int32),
        start=np.array(start_s, dtype=np.float32),
        window=np.float32(args.window),
    )
    (ROOT / "data" / "tracks.json").write_text(json.dumps(tracks, indent=1, ensure_ascii=False))
    print(f"{len(tracks)} tracks, {len(emb)} windows -> data/{tag}.npz")


if __name__ == "__main__":
    main()
