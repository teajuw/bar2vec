"""Run bar2vec on the demo songs: cut each one into sections and name them, for the demo on the article page.

Usage: uv run scripts/demo_songs.py                 the six featured songs -> article/demo/sections.json
       uv run scripts/demo_songs.py --catalogue     the searchable catalogue -> article/demo/catalogue.json

Reads article/demo/songs.json (YouTube video id, title, artist). The songs were chosen before any output
was looked at: six well-known songs across genres, none in the Harmonix Set, each an official upload
that allows embedding. The audio of each video is fetched into audio_demo/ (not in the repo) for analysis
only, so the section times line up with the video the demo page plays; the page never hosts audio.
The pipeline is the playlist one: Beat This! bars, MATPAC++ per bar, the cutter in cuts.json, and the
trained namer with per-bar CLAP and loudness: the section classifier of name_model.py with per-bar votes added
to its inputs (nm_cuts_fit.py, regime "stack", saved as data/name_model_cuts.joblib). A name below 0.8
probability is unsure. --namer v7 gives the earlier classifier without the votes.
Writes article/demo/sections.json.

With --catalogue it reads data/catalogue_songs.json instead (the chart songs matched to official uploads by
catalogue_list.py and catalogue_match.py), skips the six featured songs, and writes article/demo/catalogue.json.
Each song's bars and per-bar embeddings are kept in data/catalogue_cache/<video>.npz (not in the repo), so a
rerun continues where it stopped and a change of namer needs no audio. The fetched audio is deleted as soon as
the song is done. A song whose audio cannot be fetched is left out and listed.
"""
import argparse
import contextlib
import io
import json
import os
import time
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
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "format": "bestaudio[ext=m4a]/bestaudio", "outtmpl": str(AUDIO / "%(id)s.%(ext)s"), "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return Path(ydl.prepare_filename(ydl.extract_info(f"https://www.youtube.com/watch?v={video}", download=True)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--namer", default="stack", choices=["stack", "v7"])
    args = ap.parse_args()
    featured = json.loads((DEMO / "songs.json").read_text())
    songs = featured
    cache = ROOT / "data" / "catalogue_cache"
    if args.catalogue:
        skip = {s["video"] for s in featured}
        songs = [s for s in json.loads((ROOT / "data" / "catalogue_songs.json").read_text()) if s["video"] not in skip][: args.limit or None]
        cache.mkdir(exist_ok=True)
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
    import joblib

    import nm_cuts_fit

    model = joblib.load(ROOT / "data" / "name_model_cuts.joblib") if args.namer == "stack" else name_model.load(True)
    out, failed, last3 = [], [], []

    def named(song, db, end, m, c, loud):
        """Cut and name one song from its bars and per-bar embeddings."""
        end, m, c = float(end), np.asarray(m, dtype=np.float32), np.asarray(c, dtype=np.float32)
        starts = db.astype(np.float32)
        cuts, _ = find_cuts({ccfg["models"][0]: m}, starts, *geometry(ccfg, starts), ccfg)
        cuts = np.sort(cuts[(cuts > 0) & (cuts < end)])
        bounds = np.concatenate([[0.0], cuts, [end]])
        by_model = {"matpacbar_fold": m, "clapbar_fold": c, "loudbar_fold": loud}
        feats = name_model.raw_features(bounds, starts, by_model, text)
        if args.namer == "stack":
            bars = {"starts": starts, "f_bar": nm_cuts_fit.bar_features(starts, end, by_model, text)}
            feats = nm_cuts_fit.with_votes(feats, nm_cuts_fit.bar_votes(model, bars, bounds))
        proba = model["clf"].predict_proba(name_model.design(*feats, model))
        roles = [N.LABELS[int(model["clf"].classes_[i])] for i in proba.argmax(1)]
        groups = N.repeat_groups(N.section_vectors(bounds, starts, m), ncfg["structure"]["repeat_tau"])
        order: dict[int, int] = {}
        for g in groups:
            order.setdefault(int(g), len(order))
        result = {"duration": round(end, 2), "bars": int(len(db)), "sections": [
            {"start": round(float(a), 2), "end": round(float(b), 2), "name": r, "group": order[int(g)], **({"unsure": True} if p < 0.8 else {})}
            for a, b, r, g, p in zip(bounds[:-1], bounds[1:], roles, groups, proba.max(1))]}
        out.append({**song, **result})
        print(f"{song['artist']} - {song['title']}: {end:.0f} s, {len(db)} bars, median bar {np.median(np.diff(db)):.2f} s", flush=True)
        if not args.catalogue:
            for s in out[-1]["sections"]:
                print(f"   {int(s['start']) // 60}:{int(s['start']) % 60:02d}  {s['name']}{'?' if s.get('unsure') else ''}  (group {s['group']})")

    for song in songs:
        kept = (cache if args.catalogue else ROOT / "audio_demo") / f"{song['video']}.npz"
        if kept.exists():
            named(song, **np.load(kept))
            continue
        if args.catalogue and len(failed) >= 3 and failed[-3:] == last3:
            print("YouTube is refusing requests; stopping here. Run again later to continue.")
            break
        try:
            if args.catalogue:
                time.sleep(1.5)  # space the requests out
            path = fetch(song["video"])
            last3 = []
        except Exception as e:
            if not args.catalogue:
                raise
            failed.append(f"{song['artist']} - {song['title']}: {str(e)[:100]}")
            last3 = failed[-3:] if len(last3) >= 2 else last3 + [failed[-1]]
            continue
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

        np.savez_compressed(kept, db=db, end=end, m=m.astype(np.float16), c=c.astype(np.float16), loud=loud)
        if args.catalogue:
            path.unlink()  # the audio was only needed for the analysis
        named(song, **np.load(kept))  # from the saved copy, so a rerun gives the same result
    if args.catalogue:
        out.sort(key=lambda s: (s["artist"].lower(), s["title"].lower()))
        slim = [{k: v for k, v in s.items() if k != "kind"} for s in out]
        (DEMO / "catalogue.json").write_text(json.dumps(slim, separators=(",", ":"), ensure_ascii=False) + "\n")
        print(f"{len(out)} songs in the catalogue; {len(failed)} could not be fetched: {failed}")
        return
    (DEMO / "sections.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
