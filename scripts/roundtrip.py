"""Round-trip test for the readout: describe a window in English, search with that description,
and see whether the search leads back to the same place.

Usage: uv run scripts/roundtrip.py [--sample 300] [-n 4]

Runs every describer/searcher pair. Same-model pairs only show that a model agrees with itself.
Cross-model pairs (describe with one, search with the other) are the stronger check, because the
two models were trained separately. Neither proves a description is true; that needs listening.
"""
import argparse

import numpy as np

from readout import Readout
from search import embed_text

INDEXES = ["clap_w10_h5", "muq_w10_h5"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r = {name: Readout(name) for name in INDEXES}
    ix0 = r[INDEXES[0]].ix
    n_tracks = len(ix0.tracks)
    picks = np.random.default_rng(args.seed).choice(len(ix0.emb), args.sample, replace=False)

    print(f"{args.sample} random windows, described with their top {args.n} descriptions; {n_tracks} songs in the library")
    print(f"chance: right song first {1 / n_tracks:.0%}, in top 5 {5 / n_tracks:.0%}\n")
    print(f"{'describe with':<14}{'search with':<14}{'song first':>11}{'song top 5':>11}{'median song rank':>18}{'moment in top 10':>18}")
    for d in INDEXES:
        texts = [", ".join(p for p, _ in r[d].top(r[d].z[w], args.n)) for w in picks]
        for s in INDEXES:
            ix = r[s].ix
            q = np.concatenate([embed_text(texts[i : i + 64], s) for i in range(0, len(texts), 64)])
            scores = ix.emb @ q.T  # (windows, samples)
            per_track = np.stack([scores[ix.track == t].max(0) for t in range(n_tracks)])  # (tracks, samples)
            ranks, near = [], []
            for j, w in enumerate(picks):
                tid = ix.track[w]
                ranks.append(1 + int((per_track[:, j] > per_track[tid, j]).sum()))
                top10 = np.argsort(-scores[:, j])[:10]
                near.append(bool(np.any((ix.track[top10] == tid) & (np.abs(ix.start[top10] - ix.start[w]) <= ix.window))))
            ranks = np.array(ranks)
            print(f"{d.split('_')[0]:<14}{s.split('_')[0]:<14}{np.mean(ranks == 1):>11.0%}{np.mean(ranks <= 5):>11.0%}"
                  f"{np.median(ranks):>18.0f}{np.mean(near):>18.0%}")


if __name__ == "__main__":
    main()
