"""Break every song into sections using only its window embeddings, and describe each section.

Usage: uv run scripts/breakdown.py [--index clap_w10_h5] [--show "self care"]

For one song:
  1. Compare every window with every other window (after removing the song's average, so what is
     left is how each window differs from the rest of the song).
  2. Walk along the song. At each point, compare the few windows before with the few windows after.
     A boundary is where the two sides are each consistent but unlike each other.
  3. Give sections that resemble each other the same letter (A, B, ...), so repeats show up.
  4. Describe each section with the readout: what stands out there versus the rest of the song.

Boundaries are accurate to about five seconds at best, since windows start every five seconds.
Writes data/breakdown_<index>.json. Nothing here is trained.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from readout import Readout
from search import fmt

ROOT = Path(__file__).resolve().parent.parent


def centred(e: np.ndarray) -> np.ndarray:
    c = e - e.mean(0)
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)


def novelty(c: np.ndarray, side: int) -> np.ndarray:
    """How strongly the windows before index i differ from the windows from i onward."""
    s = c @ c.T
    n = len(s)
    nov = np.zeros(n)
    for i in range(2, n - 1):
        a, b = slice(max(0, i - side), i), slice(i, min(n, i + side))
        nov[i] = (s[a, a].mean() + s[b, b].mean()) / 2 - s[a, b].mean()
    return nov


def pick_peaks(nov: np.ndarray, min_gap: int, k: float) -> list[int]:
    cut = nov[nov > 0].mean() + k * nov[nov > 0].std() if (nov > 0).any() else np.inf
    cand = [i for i in range(1, len(nov) - 1) if nov[i] >= cut and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    chosen: list[int] = []
    for i in sorted(cand, key=lambda i: -nov[i]):
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def letters(vecs: np.ndarray, same: float) -> list[str]:
    """Greedy grouping: a section joins the first earlier group it resembles, else starts a new one."""
    groups: list[np.ndarray] = []
    out = []
    for v in vecs:
        sims = [float(v @ (g / np.linalg.norm(g))) for g in groups]
        if sims and max(sims) >= same:
            j = int(np.argmax(sims))
            groups[j] = groups[j] + v
        else:
            groups.append(v.copy())
            j = len(groups) - 1
        out.append(chr(ord("A") + min(j, 25)))
    return out


def break_down(r: Readout, tid: int, side: int = 3, min_gap: int = 3, k: float = 0.0, same: float = 0.35) -> list[dict]:
    ix = r.ix
    own = np.flatnonzero(ix.track == tid)
    starts, _, zc = r.window_rows(tid)
    c = centred(ix.emb[own])
    cuts = [0] + pick_peaks(novelty(c, side), min_gap, k) + [len(own)]
    end_of_song = float(ix.tracks[tid].get("duration_s") or starts[-1] + ix.window)
    spans = list(zip(cuts[:-1], cuts[1:]))
    vecs = np.stack([c[a:b].mean(0) for a, b in spans])
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    names = letters(vecs, same)
    out = []
    for (a, b), name in zip(spans, names):
        out.append(
            {
                "start": float(starts[a]),
                "end": float(starts[b]) if b < len(own) else end_of_song,
                "letter": name,
                "says": [p for p, _ in r.top(zc[a:b].mean(0), 3, skip=("genre",))],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="clap_w10_h5")
    ap.add_argument("--show", help="print the breakdown of one song")
    args = ap.parse_args()

    r = Readout(args.index)
    result = {t["id"]: break_down(r, t["id"]) for t in r.ix.tracks}
    (ROOT / "data" / f"breakdown_{args.index}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    n = [len(v) for v in result.values()]
    repeats = np.mean([len(v) - len({s["letter"] for s in v}) > 0 for v in result.values()])
    print(f"{args.index}: {len(result)} songs, sections per song median {np.median(n):.0f} (min {min(n)}, max {max(n)}), "
          f"{repeats:.0%} of songs have a repeated section")
    if args.show:
        tid = next(t["id"] for t in r.ix.tracks if args.show.lower() in t["title"].lower())
        print(f"\n{r.ix.tracks[tid]['title']}")
        for s in result[tid]:
            print(f"  {fmt(s['start']):>5}–{fmt(s['end']):<5} {s['letter']}  {'; '.join(s['says'])}")


if __name__ == "__main__":
    main()
