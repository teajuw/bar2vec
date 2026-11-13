"""Draw the article's "bars coloured by sound" figure for one demo song.

Usage: uv run scripts/make_bars_figure.py [--video aeDdS9aIpck]   (after scripts/demo_songs.py)

Top row: every bar of the song as one cell, coloured by its MATPAC++ embedding alone. The song's mean
embedding is removed, the three directions that vary most (principal components) become red, green
and blue, and each is stretched to its 2nd-98th percentile. Nothing else goes into the colour: no cut,
no name, no annotation. Bottom row: the sections and names in article/demo/sections.json, in the demo's
colours. Writes article/bars.svg (400 wide, so its text stays readable at phone width).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from embed_clap import decode
from hx_embed_matpac import SR, load_model

ROOT = Path(__file__).resolve().parent.parent
FILL = {"intro": "#dbd8cd", "verse": "#8fc7e8", "prechorus": "#b9a6ea", "chorus": "#f2a65a", "postchorus": "#f3c98b",
        "bridge": "#7fcfa8", "outro": "#b4b1a5", "instrumental": "#e79ab0"}
SHORT = {"prechorus": "pre", "postchorus": "post", "instrumental": "inst"}
W, X0, X1 = 400, 16, 384


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
    model = load_model(dev)
    with torch.no_grad():
        emb = np.concatenate([model(torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(dev))[0].float().cpu().numpy()
                              for s, e in zip(edges[:-1], edges[1:])])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    centred = emb - emb.mean(0)
    u, sv, _ = np.linalg.svd(centred, full_matrices=False)
    pc = u[:, :3] * sv[:3]
    lo, hi = np.percentile(pc, 2, 0), np.percentile(pc, 98, 0)
    rgb = (255 * np.clip((pc - lo) / (hi - lo), 0, 1)).astype(int)
    kept = float((sv[:3] ** 2).sum() / (sv ** 2).sum())

    x = lambda t: X0 + (X1 - X0) * t / end
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 196" font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif" font-size="12">',
         f"<title>The {len(db)} bars of one song coloured by their embeddings, above the sections bar2vec found</title>",
         '<defs><pattern id="unsure" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<rect width="3" height="6" fill="#ffffff" opacity="0.55"/></pattern></defs>',
         f'<rect width="{W}" height="196" rx="10" fill="#fafaf8"/>',
         f'<text x="{X0}" y="24" font-size="14" font-weight="600" fill="#1c1c1a">{song["title"]}: {len(db)} bars, coloured by sound</text>',
         f'<text x="{X0}" y="48" fill="#55554f">Each bar\'s embedding, shown as a colour</text>']
    for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        o.append(f'<rect x="{x(s):.2f}" y="56" width="{max(x(e) - x(s) - 0.6, 0.6):.2f}" height="40" fill="#{rgb[i][0]:02x}{rgb[i][1]:02x}{rgb[i][2]:02x}"/>')
    o.append(f'<text x="{X0}" y="122" fill="#55554f">The sections bar2vec found</text>')
    for sec in song["sections"]:
        a, b = x(sec["start"]), x(sec["end"])
        o.append(f'<rect x="{a + 0.75:.2f}" y="130" width="{b - a - 1.5:.2f}" height="30" rx="2" fill="{FILL[sec["name"]]}"/>')
        if sec.get("unsure"):
            o.append(f'<rect x="{a + 0.75:.2f}" y="130" width="{b - a - 1.5:.2f}" height="30" rx="2" fill="url(#unsure)"/>')
        label = SHORT.get(sec["name"], sec["name"]) + ("?" if sec.get("unsure") else "")
        if len(label) * 6.2 < b - a - 4:
            o.append(f'<text x="{(a + b) / 2:.1f}" y="149" text-anchor="middle" font-size="11" fill="#1c1c1a">{label}</text>')
        if sec["start"] > 0:
            o.append(f'<line x1="{a:.2f}" y1="52" x2="{a:.2f}" y2="102" stroke="#1c1c1a" stroke-width="0.75"/>')
    o += [f'<text x="{X0}" y="180" fill="#55554f">0:00</text>', f'<text x="{X1}" y="180" text-anchor="end" fill="#55554f">{clock(end)}</text>', "</svg>"]
    (ROOT / "article" / "bars.svg").write_text("\n".join(o) + "\n")
    print(f"{song['title']}: {len(db)} bars, {emb.shape[1]} numbers per bar; the three directions shown keep {kept:.0%} of the variation")


if __name__ == "__main__":
    main()
