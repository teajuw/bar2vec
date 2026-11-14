"""Export the real numbers behind the "follow one bar" animation for one demo song.

Usage: uv run scripts/bar_story_data.py OUT.json [--video aeDdS9aIpck] [--bar N]

Follows one bar (default: the first bar of the first chorus) through the same steps as
scripts/make_process_figure.py: its sound, its log-mel spectrogram (80 bands, as the audio model
uses), its MATPAC++ embedding (all 3,840 numbers), the three principal-component values that become
its colour, and its similarity to every other bar. Also writes every bar's colour and the sections.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

from embed_clap import decode
from hx_embed_matpac import SR, load_model

ROOT = Path(__file__).resolve().parent.parent


def b64(values: np.ndarray) -> str:
    """0..1 floats as base64 bytes."""
    return base64.b64encode((255 * np.clip(values, 0, 1)).astype(np.uint8).tobytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--video", default="aeDdS9aIpck")
    ap.add_argument("--bar", type=int, default=-1)
    args = ap.parse_args()
    from beat_this.inference import Audio2Beats

    song = next(s for s in json.loads((ROOT / "article" / "demo" / "sections.json").read_text()) if s["video"] == args.video)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    audio = decode(sorted((ROOT / "audio_demo").glob(args.video + ".*"))[0], SR)
    end = len(audio) / SR
    _, downbeats = Audio2Beats(checkpoint_path="final0", device=dev, dbn=False)(audio, SR)
    db = np.asarray(downbeats, dtype=float)
    db = db[db < end - 0.25]
    edges = np.concatenate([db, [end]])
    n = len(db)
    model = load_model(dev)
    with torch.no_grad():
        emb = np.concatenate([model(torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(dev))[0].float().cpu().numpy()
                              for s, e in zip(edges[:-1], edges[1:])])
    raw = emb.copy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    centred = emb - emb.mean(0)
    u, sv, _ = np.linalg.svd(centred, full_matrices=False)
    pc = u[:, :3] * sv[:3]
    pc *= np.where(pc[: n // 4].mean(0) < 0, -1, 1)
    lo, hi = np.percentile(pc, 2, 0), np.percentile(pc, 98, 0)
    scaled = np.clip((pc - lo) / (hi - lo), 0, 1)
    rgb = (255 * scaled).astype(int)
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True)

    k = args.bar
    if k < 0:
        chorus = next(s for s in song["sections"] if s["name"] == "chorus")
        k = int(np.argmax(db >= chorus["start"] - 0.05))
    a, b = int(edges[k] * SR), int(edges[k + 1] * SR)
    clip = audio[a:b]
    wave = np.array([np.abs(c).max() for c in np.array_split(clip, 160)])
    mel = torch.log(torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=160, n_mels=80)(torch.from_numpy(clip.copy())) + 1e-5).numpy()
    cols = 16 * round(mel.shape[1] / 16)  # a whole number of 16-wide patches
    spec = mel[:, np.linspace(0, mel.shape[1] - 1, cols).round().astype(int)][::-1]  # nearest frame for each column
    spec = (spec - np.percentile(spec, 5)) / (np.percentile(spec, 99.5) - np.percentile(spec, 5))
    peaks = np.array([np.abs(c).max() for c in np.array_split(audio, 240)])
    v = raw[k]
    span = np.percentile(np.abs(v), 99)
    out = {
        "title": song["title"], "artist": song.get("artist", ""), "duration": end, "bars": n, "dims": int(emb.shape[1]),
        "kept": float((sv[:3] ** 2).sum() / (sv ** 2).sum()),
        "edges": [round(float(t), 3) for t in edges],
        "peaks": b64(peaks / peaks.max()),
        "rgb": rgb.tolist(),
        "sections": [{"name": s["name"], "start": s["start"], "end": s["end"], "unsure": bool(s.get("unsure"))} for s in song["sections"]],
        "bar": {
            "index": k, "start": float(edges[k]), "end": float(edges[k + 1]),
            "wave": b64(wave / wave.max()),
            "spec": b64(spec), "specRows": int(spec.shape[0]), "specCols": int(spec.shape[1]),
            "emb": b64(0.5 + 0.5 * v / span),  # 0.5 is zero; the ends are the 99th-percentile size
            "first": [round(float(x), 3) for x in v[:8]],
            "pc": [round(float(x), 3) for x in pc[k]], "scaled": [round(float(x), 3) for x in scaled[k]],
            "sim": [round(float(x), 3) for x in unit @ unit[k]],
        },
    }
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    top = np.argsort(-(unit @ unit[k]))[1:9]
    print(f"{song['title']}: bar {k + 1} of {n} ({edges[k]:.1f}-{edges[k + 1]:.1f}s), spec {spec.shape}, rgb {rgb[k].tolist()}, "
          f"nearest bars {sorted((top + 1).tolist())}, {Path(args.out).stat().st_size // 1024} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
