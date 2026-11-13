"""Embed each song one bar at a time with CLAP (bar-aligned windows instead of a fixed grid).

Usage: uv run scripts/hx_embed_bars.py --downbeats annotated [--split-flow song-cuts]

A bar runs from one downbeat to the next (about 2 s). CLAP expects 10 s of audio, so each bar is
repeated to fill 10 s, which is also what the library does with short clips.

--downbeats annotated uses the Harmonix human downbeat annotations. That is an ORACLE condition:
it shows what bar alignment can do with perfect bars, and is not something available for an
arbitrary song. A real pipeline needs a beat tracker to estimate downbeats.
Writes data/hx_clapbar_<downbeats>.npz (start = bar start in seconds, window = median bar length).
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

from embed_clap import CKPT, decode

ROOT = Path(__file__).resolve().parent.parent
SR, FILL = 48000, 10 * 48000
BEATS = ROOT / "external" / "harmonixset" / "dataset" / "beats_and_downbeats"


def annotated_downbeats(name: str) -> np.ndarray:
    rows = [line.split() for line in (BEATS / f"{name}.txt").read_text().splitlines() if line.strip()]
    return np.array([float(r[0]) for r in rows if int(float(r[1])) == 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downbeats", default="annotated", choices=["annotated", "estimated", "fold"])
    ap.add_argument("--split-flow", default="song-cuts", help="embed only songs in this flow's split ('' for all)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--suffix", default="", help="added to the output file name, e.g. _final")
    ap.add_argument("--tracks", default="hx_tracks.json", help="song list under data/")
    args = ap.parse_args()

    import laion_clap

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=dev)
    with contextlib.redirect_stdout(io.StringIO()):
        model.load_ckpt(str(CKPT))

    tracks = json.loads((ROOT / "data" / args.tracks).read_text())
    if args.split_flow:
        st = json.loads((ROOT / ".claude" / "hillclimb" / args.split_flow / "_state.json").read_text())
        keep = set(st["train_ids"]) | set(st["test_ids"])
        tracks = [t for t in tracks if t["name"] in keep]

    estimated = {}
    if args.downbeats != "annotated":
        src = "hx_downbeats_est.json" if args.downbeats == "estimated" else "hx_downbeats_fold.json"
        estimated = {k: np.array(v) for k, v in json.loads((ROOT / "data" / src).read_text()).items()}
    embs, track_ids, start_s, bar_len = [], [], [], []
    for t in tqdm(tracks, desc="bars"):
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        db = annotated_downbeats(t["name"]) if args.downbeats == "annotated" else estimated[t["name"]]
        db = db[db < len(audio) / SR - 0.25]
        edges = np.concatenate([db, [len(audio) / SR]])
        chunks = []
        for s, e in zip(edges[:-1], edges[1:]):
            bar = audio[int(s * SR) : int(e * SR)]
            chunks.append(np.tile(bar, int(np.ceil(FILL / max(len(bar), 1))))[:FILL])
        chunks = np.stack(chunks)
        for b in range(0, len(chunks), args.batch):
            embs.append(np.asarray(model.get_audio_embedding_from_data(x=chunks[b : b + args.batch], use_tensor=False), dtype=np.float32))
        track_ids += [t["id"]] * len(db)
        start_s += [float(x) for x in db]
        bar_len.append(float(np.median(np.diff(db))))

    emb = np.concatenate(embs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = ROOT / "data" / f"hx_clapbar_{args.downbeats}{args.suffix}.npz"
    np.savez(out, emb=emb, track=np.array(track_ids, dtype=np.int32), start=np.array(start_s, dtype=np.float32),
             window=np.float32(np.median(bar_len)))
    print(f"{len(tracks)} tracks, {len(emb)} bars -> {out.name}")


if __name__ == "__main__":
    main()
