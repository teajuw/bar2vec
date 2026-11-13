"""Probe: how much does a CLAP embedding move when the same audio is shifted in pitch?

Usage: uv run scripts/probe_pitch.py [--songs 15] [--clips 4]

For random 10 s clips from the tuning songs, compares the original clip's embedding with:
  same clip, 1 s later            almost the same audio (how similar "the same thing" looks)
  shifted up 0.1 semitone         pitch unchanged to the ear, one pass through the pitch-shifter (artifact control)
  shifted up 2 then down 2        same pitch, two passes through the pitch-shifter
  shifted up 1 / 2 / 5 semitones  same performance in a different key
  another clip, same song         different part of the same song
  a clip from another song        unrelated audio
Reports mean cosine similarity for each. If a pitch shift looks like the artifact control, CLAP is
close to deaf to key; if it looks like "another clip, same song", CLAP hears key changes clearly.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import numpy as np
import torch

from embed_clap import CKPT, decode

ROOT = Path(__file__).resolve().parent.parent
SR, WIN = 48000, 480000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=15)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--model", default="clap", choices=["clap", "muq"])
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.model == "clap":
        import laion_clap

        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=dev)
        with contextlib.redirect_stdout(io.StringIO()):
            model.load_ckpt(str(CKPT))
        run = lambda x: np.asarray(model.get_audio_embedding_from_data(x=x, use_tensor=False), dtype=np.float32)
    else:  # MuQ-MuLan takes 24 kHz audio; the same clips are resampled after any pitch shift
        from embed_muq import load_model

        model = load_model(dev)

        def run(x: np.ndarray) -> np.ndarray:
            y = librosa.resample(x, orig_sr=SR, target_sr=24000, axis=-1).astype(np.float32)
            with torch.no_grad():
                return model.mulan_module.get_audio_latents(torch.from_numpy(y).to(dev)).float().cpu().numpy()

    def embed(x: np.ndarray) -> np.ndarray:
        e = np.concatenate([run(x[i : i + 16]) for i in range(0, len(x), 16)])
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    st = json.loads((ROOT / ".claude" / "hillclimb" / "song-cuts" / "_state.json").read_text())
    by_name = {t["name"]: t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text())}
    rng = np.random.default_rng(0)
    names = list(rng.choice(sorted(st["train_ids"]), args.songs, replace=False))

    shift = lambda y, n: librosa.effects.pitch_shift(y, sr=SR, n_steps=n)[:WIN]
    variants: dict[str, list[np.ndarray]] = {k: [] for k in
        ("original", "1 s later", "up 0.1 semitone (one pass)", "up 2 then down 2", "up 1 semitone", "up 2 semitones", "up 5 semitones", "another clip, same song")}
    for n in names:
        t = by_name[n]
        audio = decode(ROOT / t["file"], SR)[int(max(0, t["offset_s"]) * SR):][: int(t["ref_duration"] * SR)]
        starts = np.linspace(20 * SR, len(audio) - WIN - 2 * SR, args.clips + 1).astype(int)
        for i, s in enumerate(starts[:-1]):
            y = audio[s : s + WIN]
            variants["original"].append(y)
            variants["1 s later"].append(audio[s + SR : s + SR + WIN])
            variants["up 0.1 semitone (one pass)"].append(shift(y, 0.1))
            variants["up 2 then down 2"].append(shift(shift(y, 2), -2))
            for k, steps in (("up 1 semitone", 1), ("up 2 semitones", 2), ("up 5 semitones", 5)):
                variants[k].append(shift(y, steps))
            other = starts[(i + args.clips // 2 + 1) % (args.clips + 1)]
            variants["another clip, same song"].append(audio[other : other + WIN])
    emb = {k: embed(np.stack(v)) for k, v in variants.items()}
    o = emb["original"]
    n_clips = len(o)
    print(f"model: {args.model}")
    print(f"{len(names)} songs, {n_clips} clips; mean cosine similarity to the original clip (1.0 = identical)\n")
    for k in list(variants)[1:]:
        sims = np.sum(o * emb[k], axis=1)
        print(f"  {k:<30}{sims.mean():.3f}  (spread {sims.std():.3f})")
    perm = (np.arange(n_clips) + args.clips * (len(names) // 2)) % n_clips  # a clip from a different song
    sims = np.sum(o * o[perm], axis=1)
    print(f"  {'a clip from another song':<30}{sims.mean():.3f}  (spread {sims.std():.3f})")


if __name__ == "__main__":
    main()
