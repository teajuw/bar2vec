"""Run bar2vec on the demo songs: cut each one into sections and name them, for the demo on the article page.

Usage: uv run scripts/demo_songs.py

Reads article/demo/songs.json (YouTube video id, title, artist). The songs were chosen before any output
was looked at: six well-known songs across genres, none in the Harmonix Set, each an official upload
that allows embedding. The audio of each video is fetched into audio_demo/ (not in the repo) for analysis
only, so the section times line up with the video the demo page plays; the page never hosts audio.
The pipeline is the playlist one: Beat This! bars, MATPAC++ per bar, the cutter in cuts.json, and the
trained namer (name_model.py) with per-bar CLAP and loudness. A name below 0.8 probability is unsure.
Writes article/demo/sections.json.
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

import name_model
import names as N
import nm_run
from cuts import find_cuts, geometry
from embed_clap import CKPT, decode
from hx_embed_matpac import SR, load_model

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "article" / "demo"
AUDIO = ROOT / "audio_demo"
CSR, FILL = 48000, 10 * 48000
LSR, HOP = 22050, 512


def fetch(video: str) -> Path:
    have = sorted(AUDIO.glob(f"{video}.*"))
    if have:
        return have[0]
    import yt_dlp

    AUDIO.mkdir(exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "format": "bestaudio[ext=m4a]/bestaudio", "outtmpl": str(AUDIO / "%(id)s.%(ext)s"), "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return Path(ydl.prepare_filename(ydl.extract_info(f"https://www.youtube.com/watch?v={video}", download=True)))


def main() -> None:
    import laion_clap
    from beat_this.inference import Audio2Beats

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tracker = Audio2Beats(checkpoint_path="final0", device=dev, dbn=False)
    matpac = load_model(dev)
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=dev)
    with contextlib.redirect_stdout(io.StringIO()):
        clap.load_ckpt(str(CKPT))
    mel = torchaudio.transforms.MelSpectrogram(LSR, n_fft=1024, hop_length=HOP, n_mels=64)
    ccfg = json.loads((ROOT / "cuts.json").read_text())
    ncfg = json.loads((ROOT / "names.json").read_text())
    text = nm_run.text_vectors(ncfg["encoder"], ncfg["extra_prompts"])
    model = name_model.load(True)
    out = []
    for song in json.loads((DEMO / "songs.json").read_text()):
        path = fetch(song["video"])
        audio = decode(path, SR)
        end = len(audio) / SR
        _, downbeats = tracker(audio, SR)
        db = np.asarray(downbeats, dtype=float)
        db = db[db < end - 0.25]
        edges = np.concatenate([db, [end]])
        with torch.no_grad():  # MATPAC++ per bar, as in playlist_bars.py
            m = np.concatenate([matpac(torch.from_numpy(audio[int(s * SR) : int(e * SR)].copy())[None].to(dev))[0].float().cpu().numpy()
                                for s, e in zip(edges[:-1], edges[1:])])
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        hi = decode(path, CSR)  # CLAP per bar, each bar repeated to fill 10 s, as in hx_embed_bars.py
        chunks = np.stack([np.tile(hi[int(s * CSR) : int(e * CSR)], int(np.ceil(FILL / max(int(e * CSR) - int(s * CSR), 1))))[:FILL]
                           for s, e in zip(edges[:-1], edges[1:])])
        c = np.concatenate([np.asarray(clap.get_audio_embedding_from_data(x=chunks[b : b + 32], use_tensor=False), dtype=np.float32)
                            for b in range(0, len(chunks), 32)])
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        low = decode(path, LSR)  # loudness per bar, as in hx_loud_bars.py
        frames = torch.log(mel(torch.from_numpy(low.copy())) + 1e-5).numpy().mean(0)
        loud = np.array([frames[int(s * LSR / HOP) : max(int(e * LSR / HOP), int(s * LSR / HOP) + 1)].mean() for s, e in zip(edges[:-1], edges[1:])],
                        dtype=np.float32)[:, None]

        starts = db.astype(np.float32)
        cuts, _ = find_cuts({ccfg["models"][0]: m}, starts, *geometry(ccfg, starts), ccfg)
        cuts = np.sort(cuts[(cuts > 0) & (cuts < end)])
        bounds = np.concatenate([[0.0], cuts, [end]])
        by_model = {"matpacbar_fold": m, "clapbar_fold": c, "loudbar_fold": loud}
        proba = model["clf"].predict_proba(name_model.design(*name_model.raw_features(bounds, starts, by_model, text), model))
        roles = [N.LABELS[int(model["clf"].classes_[i])] for i in proba.argmax(1)]
        groups = N.repeat_groups(N.section_vectors(bounds, starts, m), ncfg["structure"]["repeat_tau"])
        order: dict[int, int] = {}
        for g in groups:
            order.setdefault(int(g), len(order))
        out.append({**song, "duration": round(end, 2), "bars": int(len(db)), "sections": [
            {"start": round(float(a), 2), "end": round(float(b), 2), "name": r, "group": order[int(g)], **({"unsure": True} if p < 0.8 else {})}
            for a, b, r, g, p in zip(bounds[:-1], bounds[1:], roles, groups, proba.max(1))]})
        print(f"{song['artist']} - {song['title']}: {end:.0f} s, {len(db)} bars, median bar {np.median(np.diff(db)):.2f} s")
        for s in out[-1]["sections"]:
            print(f"   {int(s['start']) // 60}:{int(s['start']) % 60:02d}  {s['name']}{'?' if s.get('unsure') else ''}  (group {s['group']})")
    (DEMO / "sections.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
