"""Try several values of one setting on the TRAIN songs only, to choose a round's candidate.

Usage: uv run scripts/hx_sweep.py --param offset_s --values 0 1.25 2.5 3.75 5
Values are parsed as JSON (so '["clap","muq"]' works for --param models).

Held-out songs are never scored here: the value is picked on train, then scripts/hx_run.py scores
that one candidate on every song and scripts/hx_report.py applies the keep rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cuts import find_cuts, geometry
from hx_run import FLOW, ROOT, grade, load_embeddings, reference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="other settings for this sweep only, as key=json")
    args = ap.parse_args()

    base = json.loads((ROOT / "cuts.json").read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        base[k] = json.loads(v)
    base.setdefault(args.param, None)
    train = set(json.loads((FLOW / "_state.json").read_text())["train_ids"])
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in train]
    cache: dict = {}
    print(f"{len(tracks)} train songs; current {args.param} = {json.dumps(base[args.param])}")
    print(f"{'value':<22}{'F 3s trimmed':>14}{'F 0.5s trimmed':>16}{'cuts/song':>11}")
    for raw in args.values:
        cfg = {**base, args.param: json.loads(raw)}
        scores, n = [], []
        for t in tracks:
            z = {}
            for k in cfg["models"]:
                key = (k, cfg["window"], cfg["hop"])
                if key not in cache:
                    cache[key] = load_embeddings(k, cfg)
                z[k] = cache[key]
            first = z[cfg["models"][0]]
            starts = first["start"][first["track"] == t["id"]]
            ref, _ = reference(t["name"])
            cuts, _ = find_cuts({k: v["emb"][v["track"] == t["id"]] for k, v in z.items()}, starts, *geometry(cfg, starts), cfg)
            cuts = np.sort(cuts[(cuts > 0) & (cuts < ref[-1])])
            g = grade(ref, np.concatenate([[0.0], cuts, [ref[-1]]]))
            scores.append((g["f3_trim"], g["f05_trim"]))
            n.append(len(cuts))
        m = np.mean(scores, axis=0) * 100
        print(f"{raw:<22}{m[0]:>14.1f}{m[1]:>16.1f}{np.mean(n):>11.1f}")


if __name__ == "__main__":
    main()
