"""Per-bar CLAP embeddings and loudness for the playlist songs: the extra inputs of the trained namer.

Usage: uv run scripts/playlist_bars_extra.py   (after scripts/playlist_bars.py)

Uses the bars already tracked by playlist_bars.py (data/pl_downbeats.json), so rows line up one to one
with data/pl_matpacbar.npz. CLAP is run as in hx_embed_bars.py (each bar repeated to fill 10 s) and
loudness as in hx_loud_bars.py (mean log-mel energy of the bar), the inputs the namer was fitted on.
Writes data/pl_clapbar.npz and data/pl_loudbar.npz (emb, track, start).
"""
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from embed_clap import CKPT, decode

ROOT = Path(__file__).resolve().parent.parent
SR, FILL = 48000, 10 * 48000
LSR, HOP = 22050, 512


def main() -> None:
    import laion_clap

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=dev)
    with contextlib.redirect_stdout(io.StringIO()):
        model.load_ckpt(str(CKPT))
    mel = torchaudio.transforms.MelSpectrogram(LSR, n_fft=1024, hop_length=HOP, n_mels=64)
    tracks = json.loads((ROOT / "data" / "tracks.json").read_text())
    grid = json.loads((ROOT / "data" / "pl_downbeats.json").read_text())
    ref = np.load(ROOT / "data" / "pl_matpacbar.npz")
    embs, level, track_ids, start_s = [], [], [], []
    for t in tqdm(tracks, desc="bars"):
        db = np.array(grid[str(t["id"])]["downbeats"], dtype=float)
        audio = decode(ROOT / t["file"], SR)
        edges = np.concatenate([db, [len(audio) / SR]])
        chunks = []
        for s, e in zip(edges[:-1], edges[1:]):
            bar = audio[int(s * SR) : int(e * SR)]
            chunks.append(np.tile(bar, int(np.ceil(FILL / max(len(bar), 1))))[:FILL])
        chunks = np.stack(chunks)
        for b in range(0, len(chunks), 32):
            embs.append(np.asarray(model.get_audio_embedding_from_data(x=chunks[b : b + 32], use_tensor=False), dtype=np.float32))
        low = decode(ROOT / t["file"], LSR)
        frames = torch.log(mel(torch.from_numpy(low.copy())) + 1e-5).numpy().mean(0)
        ledges = np.concatenate([db, [len(low) / LSR]])
        for s, e in zip(ledges[:-1], ledges[1:]):
            i, j = int(s * LSR / HOP), max(int(e * LSR / HOP), int(s * LSR / HOP) + 1)
            level.append(float(frames[i:j].mean()))
        track_ids += [t["id"]] * len(db)
        start_s += [float(x) for x in db]
    emb = np.concatenate(embs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    track, start = np.array(track_ids, dtype=np.int32), np.array(start_s, dtype=np.float32)
    assert np.array_equal(track, ref["track"]) and np.allclose(start, ref["start"], atol=2e-3), "bars differ from pl_matpacbar.npz"
    np.savez(ROOT / "data" / "pl_clapbar.npz", emb=emb, track=track, start=start)
    np.savez(ROOT / "data" / "pl_loudbar.npz", emb=np.array(level, dtype=np.float32)[:, None], track=track, start=start)
    print(f"{len(tracks)} songs, {len(emb)} bars; finite: {np.isfinite(emb).all() and np.isfinite(level).all()}")


if __name__ == "__main__":
    main()
