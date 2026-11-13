"""Find moments that sound like a moment already in the library (audio-to-audio, no text).

Usage: uv run scripts/similar.py "self care" 4:30 [-k 8] [--index clap_w10_h5] [--include-self]

The query is the indexed window of the named track that starts closest to the given time.
By default the query's own track is left out of the results.
"""
import argparse

import numpy as np

from search import Index, fmt, moments


def parse_time(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0]


def find_track(ix: Index, name: str) -> int:
    hits = [t for t in ix.tracks if name.lower() in t["title"].lower()]
    if not hits:
        raise SystemExit(f'no track title contains "{name}"')
    if len(hits) > 1:
        exact = [t for t in hits if t["title"].lower() == name.lower()]
        if len(exact) != 1:
            raise SystemExit("ambiguous title, matches: " + ", ".join(t["title"] for t in hits))
        hits = exact
    return hits[0]["id"]


def similar(ix: Index, tid: int, at: float, k: int = 8, include_self: bool = False) -> tuple[float, list[dict]]:
    """Return (query window start, top-k moments) for the window of track `tid` nearest `at` seconds."""
    own = np.flatnonzero(ix.track == tid)
    qi = own[np.argmin(np.abs(ix.start[own] - at))]
    keep = np.ones(len(ix.emb), dtype=bool) if include_self else ix.track != tid
    keep[qi] = False
    sub = Index(ix.emb[keep], ix.track[keep], ix.start[keep], ix.window, ix.tracks)
    return float(ix.start[qi]), moments(sub, ix.emb[qi], k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track", help="part of a track title")
    ap.add_argument("time", help="m:ss or seconds")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--index", default="clap_w10_h5")
    ap.add_argument("--include-self", action="store_true")
    args = ap.parse_args()

    ix = Index.load(args.index)
    tid = find_track(ix, args.track)
    start, ms = similar(ix, tid, parse_time(args.time), args.k, args.include_self)

    t = ix.tracks[tid]
    print(f"\nquery: {t['title']} — {t['artist']}  {fmt(start)}–{fmt(start + ix.window)}  [{args.index}]")
    for m in ms:
        mt = ix.tracks[m["track"]]
        print(f"  {m['score']:.3f}  {fmt(m['start'])}–{fmt(m['end'])}  {mt['title']} — {mt['artist']}")


if __name__ == "__main__":
    main()
