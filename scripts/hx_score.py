"""Score the song-breakdown cuts against Harmonix Set section annotations.

Usage: uv run scripts/hx_score.py [--tag w10_h5] [--side 3] [--min-gap 3] [--k 0.0]

Measure: the field's standard boundary score (mir_eval). A predicted cut counts as a hit when it
is within 0.5 s or within 3 s of an annotated one; F combines hits per prediction and hits per
annotation. "Trimmed" drops the first and last boundary of each song, which are free hits.

Rows:
  even spacing   control: the same number of cuts as the pipeline, spread evenly over the song
  <model>        the breakdown pipeline as it stands (cut time = start of the window after the change)
  <model> +half  the same cuts moved to the middle of the overlap between the two sides
"""
import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np

from breakdown import centred, novelty, pick_peaks

ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "external" / "harmonixset" / "dataset" / "segments"


def reference(name: str) -> np.ndarray:
    times = [float(line.split()[0]) for line in (SEG / f"{name}.txt").read_text().splitlines() if line.strip()]
    return np.array(sorted(set(times)))


def intervals(bounds: np.ndarray) -> np.ndarray:
    return np.stack([bounds[:-1], bounds[1:]], axis=1)


def score(ref: np.ndarray, est: np.ndarray) -> list[float]:
    out = []
    for trim in (False, True):
        for tol in (0.5, 3.0):
            if trim and (len(ref) < 3 or len(est) < 3):
                out.append(0.0)
                continue
            out.append(mir_eval.segment.detection(intervals(ref), intervals(est), window=tol, trim=trim)[2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w10_h5")
    ap.add_argument("--side", type=int, default=3)
    ap.add_argument("--min-gap", type=int, default=3)
    ap.add_argument("--k", type=float, default=0.0)
    args = ap.parse_args()
    tracks = json.loads((ROOT / "data" / "hx_tracks.json").read_text())
    rows: dict[str, list[list[float]]] = {}
    counts: dict[str, list[int]] = {}

    for kind in ("clap", "muq"):
        z = np.load(ROOT / "data" / f"hx_{kind}_{args.tag}.npz")
        hop = float(np.median(np.diff(z["start"][z["track"] == tracks[0]["id"]])))
        half = (float(z["window"]) - hop) / 2
        for t in tracks:
            ref = reference(t["name"])
            end = ref[-1]
            own = z["track"] == t["id"]
            starts = z["start"][own]
            peaks = pick_peaks(novelty(centred(z["emb"][own]), args.side), args.min_gap, args.k)
            cuts = np.array([starts[i] for i in peaks], dtype=float)
            variants = {kind: cuts, f"{kind} +half": cuts + half}
            if kind == "clap":
                variants["even spacing"] = np.linspace(0, end, len(cuts) + 2)[1:-1]
            for label, c in variants.items():
                c = c[(c > 0) & (c < end)]
                est = np.concatenate([[0.0], c, [end]])
                rows.setdefault(label, []).append(score(ref, est))
                counts.setdefault(label, []).append(len(est) - 1)
        ref_n = [len(reference(t["name"])) - 1 for t in tracks]

    print(f"{len(tracks)} songs, annotated sections per song: mean {np.mean(ref_n):.1f}\n")
    print(f"{'':<16}{'F 0.5 s':>9}{'F 3 s':>9}{'F 0.5 s trimmed':>17}{'F 3 s trimmed':>15}{'sections':>10}")
    for label in ["even spacing", "clap", "clap +half", "muq", "muq +half"]:
        m = np.mean(rows[label], axis=0) * 100
        print(f"{label:<16}{m[0]:>9.1f}{m[1]:>9.1f}{m[2]:>17.1f}{m[3]:>15.1f}{np.mean(counts[label]):>10.1f}")
    sd = np.std(rows["clap"], axis=0)[1] * 100
    print(f"\nspread across songs (standard deviation of F 3 s, clap): {sd:.1f} points")


if __name__ == "__main__":
    main()
