"""Break every playlist song into sections with the bar2vec cutter (cuts.json), one embedding per bar.

Usage: uv run scripts/breakdown_bars.py [--show "self care"]   (after scripts/playlist_bars.py)

Cuts come from scripts/cuts.py with the settings in cuts.json, the cutter tuned on the Harmonix Set:
block matching proposes cuts on bar starts and the change score vetoes weak ones. Letters are the
namer's own repeat groups (names.py), in order of first appearance, so sections that share a letter
are the ones the namer treated as the same material. Descriptions come from the CLAP window readout, averaged over the windows centred in the section.
Names (intro, verse, chorus, ...) come from the training-free rules in names.py with the settings in
names.json: position, repetition and neighbours, plus a sung-or-not score from the CLAP windows.
Writes data/breakdown_bar2vec.json in the same format as breakdown.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import names as N
import nm_run
from cuts import find_cuts, geometry
from readout import Readout
from search import fmt

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the breakdown of one song")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "cuts.json").read_text())
    assert cfg["models"] == ["matpacbar_fold"], "this script feeds MATPAC++ bar embeddings only"
    z = np.load(ROOT / "data" / "pl_matpacbar.npz")
    r = Readout("clap_w10_h5")
    ncfg = json.loads((ROOT / "names.json").read_text())
    assert not ncfg.get("trained"), "the playlist page uses the training-free namer"
    text = nm_run.text_vectors(ncfg["encoder"], ncfg["extra_prompts"])
    result = {}
    for t in r.ix.tracks:
        own = z["track"] == t["id"]
        emb, starts = z["emb"][own], z["start"][own]
        end = float(t.get("duration_s") or starts[-1] + 2.0)
        cuts, _ = find_cuts({cfg["models"][0]: emb}, starts, *geometry(cfg, starts), cfg)
        cuts = np.sort(cuts[(cuts > 0) & (cuts < end)])
        bounds = np.concatenate([[0.0], cuts, [end]])
        w_start, _, zc = r.window_rows(t["id"])
        mid = w_start + r.ix.window / 2
        says = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            win = (mid >= a) & (mid < b)
            row = zc[win].mean(0) if win.any() else zc[np.argmin(np.abs(mid - (a + b) / 2))]
            says.append([p for p, _ in r.top(row, 3, skip=("genre",))])
        # Section names: sung-or-not from the CLAP windows centred in each section, structure from the MATPAC++ bars.
        clap = N.section_vectors(bounds, mid, r.ix.emb[r.ix.track == t["id"]])
        d = (clap @ text["vocals"].T).max(1) - (clap @ text["no_vocals"].T).max(1)
        vocal = (d - np.median(d)) / (d.std() + 1e-9)
        mat = N.section_vectors(bounds, starts, emb)
        roles = N.by_structure(bounds, mat, ncfg["structure"], vocal)
        order: dict[int, int] = {}
        for k in N.repeat_groups(mat, ncfg["structure"]["repeat_tau"]):  # same grouping the namer uses
            order.setdefault(int(k), len(order))
        names = [chr(ord("A") + min(order[int(k)], 25)) for k in N.repeat_groups(mat, ncfg["structure"]["repeat_tau"])]
        result[t["id"]] = [{"start": round(float(a), 2), "end": round(float(b), 2), "letter": n, "says": s, "name": role}
                           for a, b, n, s, role in zip(bounds[:-1], bounds[1:], names, says, roles)]
        # "verse" is the leftover name. Two neighbours that sound unlike each other (different groups) and are both
        # "verse" are flagged: on the Harmonix tuning songs such a section is a verse 47% of the time, against 73% otherwise.
        segs = result[t["id"]]
        for i, sg in enumerate(segs):
            if sg["name"] == "verse" and any(0 <= j < len(segs) and segs[j]["name"] == "verse" and segs[j]["letter"] != sg["letter"]
                                             for j in (i - 1, i + 1)):
                sg["unsure"] = True
    (ROOT / "data" / "breakdown_bar2vec.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    n = [len(v) for v in result.values()]
    repeats = np.mean([len(v) - len({s["letter"] for s in v}) > 0 for v in result.values()])
    kinds = np.median([len({s["letter"] for s in v}) for v in result.values()])
    from collections import Counter
    print("unsure verses:", sum(bool(s.get("unsure")) for v in result.values() for s in v))
    print("names:", dict(Counter(s["name"] for v in result.values() for s in v).most_common()))
    print(f"bar2vec: {len(result)} songs, sections per song median {np.median(n):.0f} (min {min(n)}, max {max(n)}), "
          f"distinct letters median {kinds:.0f}, {repeats:.0%} of songs have a repeated section")
    if args.show:
        tid = next(t["id"] for t in r.ix.tracks if args.show.lower() in t["title"].lower())
        print(f"\n{r.ix.tracks[tid]['title']}")
        for s in result[tid]:
            print(f"  {fmt(s['start']):>5}–{fmt(s['end']):<5} {s['letter']}  {s['name']:<13}{'; '.join(s['says'])}")


if __name__ == "__main__":
    main()
