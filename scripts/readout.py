"""Read window embeddings back into English by comparing them with a vocabulary of descriptions.

Usage:
  uv run scripts/readout.py "self care"            # the song as a whole, then a timeline
  uv run scripts/readout.py "self care" 4:30       # one moment
  options: --index muq_w10_h5, -n 4

Scores are relative, not absolute. Audio and text vectors sit in separate regions of the space,
so raw similarities are low and bunched. Each description is z-scored across every window in the
library: "sounds like" means unusually high for this window compared with the library, and
"stands out" means unusually high compared with the rest of the same song.
Edit data/vocab.json to change the vocabulary.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from search import Index, embed_text, fmt
from similar import find_track, parse_time

ROOT = Path(__file__).resolve().parent.parent


def load_vocab() -> tuple[list[str], list[str]]:
    v = json.loads((ROOT / "data" / "vocab.json").read_text())
    cats = [c for c, ps in v.items() for _ in ps]
    phrases = [p for ps in v.values() for p in ps]
    return cats, phrases


def vocab_vectors(index: str, phrases: list[str]) -> np.ndarray:
    """Text embeddings for the vocabulary, cached per model family."""
    cache = ROOT / "data" / f"vocab_{index.split('_')[0]}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["phrases"]) == phrases:
            return z["emb"]
    emb = np.concatenate([embed_text(phrases[i : i + 64], index) for i in range(0, len(phrases), 64)])
    np.savez(cache, emb=emb, phrases=np.array(phrases))
    return emb


class Readout:
    def __init__(self, index: str = "clap_w10_h5"):
        self.ix = Index.load(index)
        self.cats, self.phrases = load_vocab()
        s = self.ix.emb @ vocab_vectors(index, self.phrases).T  # (windows, descriptions)
        self.z = (s - s.mean(0)) / s.std(0)

    def top(self, row: np.ndarray, n: int = 4, skip: tuple[str, ...] = ()) -> list[tuple[str, float]]:
        """Best description per category, then the n strongest of those."""
        best: dict[str, int] = {}
        for i, c in enumerate(self.cats):
            if c not in skip and (c not in best or row[i] > row[best[c]]):
                best[c] = i
        picks = sorted(best.values(), key=lambda i: -row[i])[:n]
        return [(self.phrases[i], float(row[i])) for i in picks]

    def track_row(self, tid: int) -> np.ndarray:
        return self.z[self.ix.track == tid].mean(0)

    def window_rows(self, tid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(starts, library-relative rows, rows relative to the rest of the song) for one track."""
        own = np.flatnonzero(self.ix.track == tid)
        z = self.z[own]
        return self.ix.start[own], z, z - z.mean(0)


def show(tags: list[tuple[str, float]]) -> str:
    return "; ".join(f"{p} ({z:+.1f})" for p, z in tags)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("time", nargs="?")
    ap.add_argument("--index", default="clap_w10_h5")
    ap.add_argument("-n", type=int, default=4)
    args = ap.parse_args()

    r = Readout(args.index)
    tid = find_track(r.ix, args.track)
    t = r.ix.tracks[tid]
    starts, z, zc = r.window_rows(tid)
    print(f"\n{t['title']} — {t['artist']}  [{args.index}]")
    print(f"whole song sounds like: {show(r.top(r.track_row(tid), args.n))}")
    if args.time:
        i = int(np.argmin(np.abs(starts - parse_time(args.time))))
        print(f"\n{fmt(starts[i])}–{fmt(starts[i] + r.ix.window)}")
        print(f"  sounds like: {show(r.top(z[i], args.n))}")
        print(f"  stands out from the rest of the song: {show(r.top(zc[i], args.n, skip=('genre',)))}")
    else:
        print("\nwhat stands out from the rest of the song, window by window")
        for i in range(0, len(starts), 2):
            print(f"  {fmt(starts[i]):>5}  {show(r.top(zc[i], 3, skip=('genre',)))}")


if __name__ == "__main__":
    main()
