"""Per-beat features, row-aligned with data/hx_matpacbeat_fold.npz (same songs, same beats, same order).

Usage: uv run scripts/hx_beat_feats.py mel      -> data/hx_melbeat_fold.npz   (no model)
       uv run scripts/hx_beat_feats.py matpac   -> data/hx_matpacbeat1_fold.npz (MATPAC++ on each beat alone)

mel: for each beat, the mean log-mel spectrum over the beat (64 bands), its loudness, and the spectral flux
(how much the spectrum rises) in the 100 ms around the beat's start. matpac: one 3840-number embedding of
the audio from this beat to the next, instead of a whole bar.
"""
import json
import sys

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from embed_clap import decode
from hx_embed_matpac import ROOT, load_model

KIND = sys.argv[1]
SR = 16000 if KIND == "matpac" else 22050
HOP = 256


def main() -> None:
    dev = "mps"
    st = json.loads((ROOT / ".claude" / "hillclimb" / "song-cuts" / "_state.json").read_text())
    keep = set(st["train_ids"]) | set(st["test_ids"])
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in keep]
    B = json.loads((ROOT / "data" / "hx_beats_fold.json").read_text())
    model = load_model(dev) if KIND == "matpac" else None
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=HOP, n_mels=64) if KIND == "mel" else None
    feats, tid, start = [], [], []
    for t in tqdm(tracks):
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        end = len(audio) / SR
        beats = np.array(B[t["name"]]["beats"])
        beats = beats[beats < end - 0.25]  # same rule as hx_embed_matpac_beats.py
        edges = np.concatenate([beats, [end]])
        if KIND == "matpac":
            with torch.no_grad():
                for s, e in zip(edges[:-1], edges[1:]):
                    emb, _ = model(torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(dev))
                    feats.append(emb.float().cpu().numpy()[0])
        else:
            m = torch.log(mel(torch.from_numpy(audio.copy())) + 1e-5).numpy()  # (64, frames)
            flux = np.concatenate([[0.0], np.maximum(0, np.diff(m, axis=1)).mean(0)])
            fr = lambda x: int(x * SR / HOP)
            for s, e in zip(edges[:-1], edges[1:]):
                seg = m[:, fr(s) : max(fr(e), fr(s) + 1)]
                w = flux[max(0, fr(s - 0.05)) : fr(s + 0.05) + 1]
                feats.append(np.concatenate([seg.mean(1), [seg.mean(), w.max() if len(w) else 0.0]]))
        tid += [t["id"]] * len(beats); start += beats.tolist()
    f = np.stack(feats).astype(np.float32)
    if KIND == "matpac":
        f = (f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)).astype(np.float16)
    ref = np.load(ROOT / "data" / "hx_matpacbeat_fold.npz")
    assert np.array_equal(ref["track"], np.array(tid)) and np.allclose(ref["start"], np.array(start, dtype=np.float32)), "rows do not line up"
    out = "hx_matpacbeat1_fold.npz" if KIND == "matpac" else "hx_melbeat_fold.npz"
    np.savez(ROOT / "data" / out, emb=f, track=np.array(tid, dtype=np.int32), start=np.array(start, dtype=np.float32))
    print(out, f.shape)


if __name__ == "__main__":
    main()
