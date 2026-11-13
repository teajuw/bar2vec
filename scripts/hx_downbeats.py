"""Estimate downbeats (bar starts) for the Harmonix songs with the Beat This! tracker.

Usage: uv run scripts/hx_downbeats.py [--split-flow song-cuts] [--device mps]

Runs on the same span of audio that gets embedded (shifted to the annotated recording's timeline),
so the times are directly comparable with the annotations. Writes data/hx_downbeats_est.json and
prints how well the estimates agree with Harmonix's human downbeat annotations (mir_eval beat
F-measure, 70 ms tolerance). The annotations are used only for that report.
"""
import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np
from tqdm import tqdm

from embed_clap import decode
from hx_embed_bars import annotated_downbeats

ROOT = Path(__file__).resolve().parent.parent
SR = 22050


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-flow", default="song-cuts")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fair", action="store_true",
                    help="Harmonix is in the tracker's training data. With --fair each song is tracked by the "
                         "cross-validation model that did not train on it (data/beat_this_harmonix_folds.tsv).")
    ap.add_argument("--tracks", default="hx_tracks.json", help="song list under data/")
    args = ap.parse_args()
    from beat_this.inference import Audio2Beats

    folds = dict(line.split() for line in (ROOT / "data" / "beat_this_harmonix_folds.tsv").read_text().splitlines() if line.strip())
    trackers: dict = {}

    def tracker(audio, sr, name):
        # A song missing from the fold list was in no fold's training data, so any fold model is fair for it.
        ckpt = f"fold{folds.get(name, '0')}" if args.fair else "final0"
        if ckpt not in trackers:
            trackers[ckpt] = Audio2Beats(checkpoint_path=ckpt, device=args.device, dbn=False)
        return trackers[ckpt](audio, sr)
    tracks = json.loads((ROOT / "data" / args.tracks).read_text())
    if args.split_flow:
        st = json.loads((ROOT / ".claude" / "hillclimb" / args.split_flow / "_state.json").read_text())
        keep = set(st["train_ids"]) | set(st["test_ids"])
        tracks = [t for t in tracks if t["name"] in keep]

    out_path = ROOT / "data" / ("hx_downbeats_fold.json" if args.fair else "hx_downbeats_est.json")
    tracks = sorted(tracks, key=lambda t: folds.get(t["name"], "9")) if args.fair else tracks
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for t in tqdm(tracks, desc="downbeats"):
        if t["name"] in out:
            continue
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        _, downbeats = tracker(audio, SR, t["name"])
        out[t["name"]] = [round(float(x), 3) for x in downbeats]
        if len(out) % 25 == 0:
            out_path.write_text(json.dumps(out))
    out_path.write_text(json.dumps(out))

    f = [mir_eval.beat.f_measure(annotated_downbeats(t["name"]), np.array(out[t["name"]]), f_measure_threshold=0.07)
         for t in tracks if len(out[t["name"]])]
    bars = [np.median(np.diff(out[t["name"]])) for t in tracks if len(out[t["name"]]) > 2]
    print(f"{len(tracks)} songs; agreement with annotated downbeats: mean F {np.mean(f):.3f}, "
          f"songs with F above 0.9: {np.mean(np.array(f) > 0.9):.0%}; median estimated bar {np.median(bars):.2f} s")


if __name__ == "__main__":
    main()
