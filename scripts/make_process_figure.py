"""Draw the article's step-by-step figure for one demo song: sound, spectrogram, bars, embeddings,
the bar-against-bar comparison, and the sections with names.

Usage: uv run scripts/make_process_figure.py [--video aeDdS9aIpck]   (after scripts/demo_songs.py)

Everything drawn is computed from the song's audio in one run:
  1. Sound: peak level in each slice of time.
  2. Spectrogram: log-mel energy, 64 bands. The audio model computes its own (80 bands) from each bar.
  3. Bars: the Beat This! bar lines.
  4. Embeddings: each bar's MATPAC++ embedding as a colour. The song's mean embedding is removed, the
     three directions that vary most (principal components) become red, green and blue, each stretched
     to its 2nd-98th percentile. The sign of each direction is fixed so reruns give the same colours.
  5. Compare: cosine similarity between every pair of bars (mean removed), dark = alike.
  6. Cut and name: the sections in article/demo/sections.json, in the demo's colours.
Writes article/process.svg (400 wide, so its text stays readable at phone width). The two pictures
(spectrogram, grid) are embedded in the SVG as small PNGs.
"""
import argparse
import base64
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
import torchaudio

from embed_clap import decode
from hx_embed_matpac import SR, load_model

ROOT = Path(__file__).resolve().parent.parent
FILL = {"intro": "#dbd8cd", "verse": "#8fc7e8", "prechorus": "#b9a6ea", "chorus": "#f2a65a", "postchorus": "#f3c98b",
        "bridge": "#7fcfa8", "outro": "#b4b1a5", "instrumental": "#e79ab0"}
SHORT = {"prechorus": "pre", "postchorus": "post", "instrumental": "inst"}
W, X0, X1 = 400, 16, 384
LO, HI = np.array([250, 250, 248.0]), np.array([12, 68, 124.0])  # the figure's background to its darkest blue
INK, SOFT = "#1c1c1a", "#55554f"


def png_uri(values: np.ndarray) -> str:
    """A (rows, cols) array of 0..1 as a PNG data URI, light for 0 and dark blue for 1."""
    rgb = np.clip(LO + (HI - LO) * values[..., None], 0, 255).astype(np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def clock(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="aeDdS9aIpck")
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
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    centred = emb - emb.mean(0)
    u, sv, _ = np.linalg.svd(centred, full_matrices=False)
    pc = u[:, :3] * sv[:3]
    pc *= np.where(pc[: n // 4].mean(0) < 0, -1, 1)  # fix each direction's sign: the song's first quarter sits on the positive side
    lo, hi = np.percentile(pc, 2, 0), np.percentile(pc, 98, 0)
    rgb = (255 * np.clip((pc - lo) / (hi - lo), 0, 1)).astype(int)
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True)
    sim = np.clip((unit @ unit.T + 0.2) / 1.2, 0, 1)

    cols = 368
    peaks = np.array([np.abs(c).max() for c in np.array_split(audio, cols // 2)])
    peaks /= peaks.max()
    mel = torch.log(torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=64)(torch.from_numpy(audio.copy())) + 1e-5).numpy()
    spec = np.stack([mel[:, c].mean(1) for c in np.array_split(np.arange(mel.shape[1]), cols * 2)], 1)[::-1]
    spec = np.clip((spec - np.percentile(spec, 5)) / (np.percentile(spec, 99.5) - np.percentile(spec, 5)), 0, 1)

    x = lambda t: X0 + (X1 - X0) * t / end
    label = lambda y, num, name, rest: (f'<text x="{X0}" y="{y}" fill="{SOFT}"><tspan font-weight="600" fill="{INK}">{num} {name}.</tspan> {rest}</text>')
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 566" font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif" font-size="12">',
         f"<title>{song['title']} step by step: sound, spectrogram, bars, embeddings, the bar-against-bar comparison, and named sections</title>",
         '<defs><pattern id="unsure" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<rect width="3" height="6" fill="#ffffff" opacity="0.55"/></pattern></defs>',
         f'<rect width="{W}" height="566" rx="10" fill="#fafaf8"/>',
         f'<text x="{X0}" y="24" font-size="14" font-weight="600" fill="{INK}">{song["title"]}, step by step</text>',
         label(46, 1, "Sound", "The song as it arrives.")]
    for i, p in enumerate(peaks):
        hgt = max(p * 15, 0.5)
        o.append(f'<rect x="{X0 + 2 * i:.1f}" y="{68 - hgt:.2f}" width="1.4" height="{2 * hgt:.2f}" fill="#8a8a83"/>')
    o += [label(104, 2, "Spectrogram", "The same sound, split by pitch."),
          f'<image x="{X0}" y="110" width="{X1 - X0}" height="48" preserveAspectRatio="none" href="{png_uri(spec)}"/>',
          label(178, 3, "Bars", f"A beat tracker finds {n} bar lines.")]
    for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        o.append(f'<rect x="{x(s):.2f}" y="184" width="{x(e) - x(s):.2f}" height="14" fill="#8a8a83" opacity="{0.4 if i % 2 else 0.18}"/>')
    # The audio model sits between the bars and their embeddings.
    o += [f'<line x1="200" y1="202" x2="200" y2="208" stroke="{SOFT}" stroke-width="1"/>',
          f'<rect x="92" y="208" width="216" height="24" rx="4" fill="#efeee8" stroke="#c9c7bd" stroke-width="0.75"/>',
          f'<text x="200" y="224" text-anchor="middle" font-size="11" fill="{INK}">frozen audio model, one bar at a time</text>',
          f'<path d="M200 232 V240 M196.5 236.5 L200 240 L203.5 236.5" fill="none" stroke="{SOFT}" stroke-width="1"/>',
          label(258, 4, "Embeddings", "Each bar's 3,840 numbers, as a colour.")]
    for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        o.append(f'<rect x="{x(s):.2f}" y="264" width="{max(x(e) - x(s) - 0.6, 0.6):.2f}" height="30" fill="#{rgb[i][0]:02x}{rgb[i][1]:02x}{rgb[i][2]:02x}"/>')
    g0, gs = 122, 156  # the grid is square: bars by bars, not seconds
    o += [label(314, 5, "Compare", "Every bar against every other. Dark is alike."),
          f'<image x="{g0}" y="322" width="{gs}" height="{gs}" preserveAspectRatio="none" style="image-rendering:pixelated" href="{png_uri(np.kron(sim, np.ones((3, 3))))}"/>']
    for sec in song["sections"][1:]:
        k = int(np.argmax(db >= sec["start"] - 0.05)) / n * gs
        o.append(f'<path d="M{g0 + k:.2f} 322 V{322 + gs} M{g0} {322 + k:.2f} H{g0 + gs}" stroke="#d85a30" stroke-width="0.6" fill="none"/>')
    o.append(label(498, 6, "Cut and name", "The squares become sections."))
    for sec in song["sections"]:
        a, b = x(sec["start"]), x(sec["end"])
        o.append(f'<rect x="{a + 0.75:.2f}" y="504" width="{b - a - 1.5:.2f}" height="30" rx="2" fill="{FILL[sec["name"]]}"/>')
        if sec.get("unsure"):
            o.append(f'<rect x="{a + 0.75:.2f}" y="504" width="{b - a - 1.5:.2f}" height="30" rx="2" fill="url(#unsure)"/>')
        text = SHORT.get(sec["name"], sec["name"]) + ("?" if sec.get("unsure") else "")
        if len(text) * 6.2 < b - a - 4:
            o.append(f'<text x="{(a + b) / 2:.1f}" y="523" text-anchor="middle" font-size="11" fill="{INK}">{text}</text>')
    o += [f'<text x="{X0}" y="552" fill="{SOFT}">0:00</text>', f'<text x="{X1}" y="552" text-anchor="end" fill="{SOFT}">{clock(end)}</text>', "</svg>"]
    out = ROOT / "article" / "process.svg"
    out.write_text("\n".join(o) + "\n")
    kept = float((sv[:3] ** 2).sum() / (sv ** 2).sum())
    print(f"{song['title']}: {n} bars, {emb.shape[1]} numbers per bar; colours keep {kept:.0%} of the variation; {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
