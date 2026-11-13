"""Score moment search against whole-track baselines on data/queries.json.

Usage: uv run scripts/evaluate.py [--index clap_w10_h5]

Track-level metric (comparable across methods): for each query, the rank of the best
expected track. Reports mean reciprocal rank (MRR) and hit@1 / hit@5.
  moment     a track's score = its best single window
  mean       a track's score = mean-of-windows vector
  one-crop   a track's score = one random window (averaged over several seeds)

This does NOT check that the timestamp is right; that needs listening.
Writes data/eval_<index>.json with the per-query moments for the listening page.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from search import Index, embed_text, fmt, moments

ROOT = Path(__file__).resolve().parent.parent


def best_rank(order: np.ndarray, expected: set[int]) -> int:
    for r, t in enumerate(order, 1):
        if int(t) in expected:
            return r
    return len(order) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="clap_w10_h5")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    ix = Index.load(args.index)
    queries = json.loads((ROOT / "data" / "queries.json").read_text())
    title_to_id = {t["title"]: t["id"] for t in ix.tracks}
    q_emb = embed_text([q["query"] for q in queries], args.index)

    n_tracks = len(ix.tracks)
    win_scores = ix.emb @ q_emb.T  # (N, Q)
    max_per_track = np.stack([win_scores[ix.track == t].max(0) for t in range(n_tracks)])  # (T, Q)
    mean_vecs = ix.mean_vectors()
    crop_vecs = [ix.one_crop_vectors(seed) for seed in range(args.seeds)]

    rows, detail = [], []
    for qi, q in enumerate(queries):
        missing = [e for e in q["expect"] if e not in title_to_id]
        expected = {title_to_id[e] for e in q["expect"] if e in title_to_id}
        if not expected:
            print(f"skip {q['id']}: none of the expected tracks are in the index ({missing})")
            continue
        r_moment = best_rank(np.argsort(-max_per_track[:, qi]), expected)
        r_mean = best_rank(np.argsort(-(mean_vecs @ q_emb[qi])), expected)
        r_crops = [best_rank(np.argsort(-(v @ q_emb[qi])), expected) for v in crop_vecs]
        rows.append((q["id"], r_moment, r_mean, r_crops))
        ms = moments(ix, q_emb[qi], k=5)
        detail.append(
            {
                **q,
                "missing": missing,
                "rank": {"moment": r_moment, "mean": r_mean, "one_crop_median": float(np.median(r_crops))},
                "moments": [
                    {**m, "title": ix.tracks[m["track"]]["title"], "artist": ix.tracks[m["track"]]["artist"],
                     "file": ix.tracks[m["track"]]["file"], "expected": m["track"] in expected}
                    for m in ms
                ],
                "mean_top": [ix.tracks[int(t)]["title"] for t in np.argsort(-(mean_vecs @ q_emb[qi]))[:5]],
            }
        )

    print(f"\n{'query':<20}{'moment':>8}{'mean':>8}{'one-crop (median of ' + str(args.seeds) + ')':>28}")
    for qid, a, b, c in rows:
        print(f"{qid:<20}{a:>8}{b:>8}{np.median(c):>28.0f}")

    def summarise(ranks: np.ndarray) -> str:
        return f"MRR {np.mean(1 / ranks):.3f}   hit@1 {np.mean(ranks <= 1):.0%}   hit@5 {np.mean(ranks <= 5):.0%}"

    a = np.array([r[1] for r in rows], dtype=float)
    b = np.array([r[2] for r in rows], dtype=float)
    c = np.array([r[3] for r in rows], dtype=float)  # (Q, seeds)
    print(f"\n{len(rows)} queries, {n_tracks} tracks")
    print(f"moment     {summarise(a)}")
    print(f"mean       {summarise(b)}")
    mrr = (1 / c).mean(0)
    print(f"one-crop   MRR {mrr.mean():.3f} (min {mrr.min():.3f}, max {mrr.max():.3f} across seeds)"
          f"   hit@1 {np.mean(c <= 1):.0%}   hit@5 {np.mean(c <= 5):.0%}")

    out = ROOT / "data" / f"eval_{args.index}.json"
    out.write_text(json.dumps(detail, indent=1, ensure_ascii=False))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
