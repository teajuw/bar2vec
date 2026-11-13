"""Per-bar loudness for the Harmonix songs: the mean log-mel energy of each bar (64 mel bands, as in
hx_beat_feats.py). Averaging in the log domain rewards a full spectrum, not only a high peak level.

Usage: uv run scripts/hx_loud_bars.py [--split-flow song-cuts] [--suffix _final]

Same bars as the per-bar embeddings (fold-held-out downbeats, same end-of-song rule), so rows line up
one to one with hx_matpacbar_fold. No model. --kind rms writes plain decibel level instead (tested, weaker). Writes data/hx_loudbar_fold<suffix>.npz (emb = one number per bar).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from embed_clap import decode

ROOT = Path(__file__).resolve().parent.parent
SR, HOP = 22050, 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-flow", default="song-cuts")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--kind", default="mel", choices=["mel", "rms"])
    ap.add_argument("--tracks", default="hx_tracks.json", help="song list under data/")
    args = ap.parse_args()
    st = json.loads((ROOT / ".claude" / "hillclimb" / args.split_flow / "_state.json").read_text())
    keep = set(st["train_ids"]) | set(st["test_ids"])
    tracks = [t for t in json.loads((ROOT / "data" / args.tracks).read_text()) if t["name"] in keep]
    downbeats = json.loads((ROOT / "data" / "hx_downbeats_fold.json").read_text())
    import torch
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=HOP, n_mels=64)
    level, track_ids, start_s = [], [], []
    for t in tqdm(tracks, desc="loudness"):
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        db = np.array(downbeats[t["name"]])
        db = db[db < len(audio) / SR - 0.25]  # same rule as hx_embed_bars.py
        edges = np.concatenate([db, [len(audio) / SR]])
        frames = torch.log(mel(torch.from_numpy(audio.copy())) + 1e-5).numpy().mean(0)  # mean log-mel per frame
        for s, e in zip(edges[:-1], edges[1:]):
            if args.kind == "rms":
                bar = audio[int(s * SR) : int(e * SR)].astype(np.float64)
                level.append(10 * np.log10(np.mean(bar ** 2) + 1e-10) if len(bar) else -100.0)
            else:
                i, j = int(s * SR / HOP), max(int(e * SR / HOP), int(s * SR / HOP) + 1)
                level.append(float(frames[i:j].mean()))
        track_ids += [t["id"]] * len(db)
        start_s += [float(x) for x in db]
    out = ROOT / "data" / f"hx_loudbar_fold{args.suffix}.npz"
    np.savez(out, emb=np.array(level, dtype=np.float32)[:, None], track=np.array(track_ids, dtype=np.int32),
             start=np.array(start_s, dtype=np.float32))
    print(f"{len(tracks)} tracks, {len(level)} bars -> {out.name}")


if __name__ == "__main__":
    main()
