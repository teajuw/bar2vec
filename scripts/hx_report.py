"""Hill-climb bookkeeping for the song-cut climb: draw the split once, then print the status table.

Usage:
  uv run scripts/hx_report.py --init          # draw the random split, write _state.json (once)
  uv run scripts/hx_report.py [--ref baseline] # status table; deltas are paired, per song, versus --ref

Every number is recomputed from the per-song rows in <variant>/results.jsonl. A delta is "beyond
noise" when the 95% interval of the per-song differences excludes zero.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / ".claude" / "hillclimb" / "song-cuts"
HARNESS = ["scripts/hx_run.py", "scripts/hx_report.py", "data/hx_tracks.json", "uv.lock"]
TARGET, GUARD = "f3_trim", "f05_trim"


def digest() -> str:
    h = hashlib.sha256()
    for p in HARNESS:
        h.update((ROOT / p).read_bytes())
    return h.hexdigest()[:16]


def rows(variant: str) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in map(json.loads, (FLOW / variant / "results.jsonl").read_text().splitlines())}


def init(seed: int, tracks_file: str, prev: str) -> None:
    tracks = json.loads((ROOT / "data" / tracks_file).read_text())
    by_genre: dict[str, list[str]] = {}
    for t in tracks:
        by_genre.setdefault(t["genre"], []).append(t["name"])
    rng = random.Random(seed)
    # Songs already assigned in an earlier, smaller split keep their side; new songs are dealt at random.
    prev_path = FLOW.parent / prev / "_state.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {"train_ids": [], "test_ids": []}
    was_train, was_test = set(prev["train_ids"]), set(prev["test_ids"])
    train, test, flip = [], [], 0
    for genre in sorted(by_genre):
        names = sorted(by_genre[genre])
        rng.shuffle(names)
        for n in names:
            if n in was_train:
                train.append(n)
            elif n in was_test:
                test.append(n)
            else:
                (train if flip % 2 == 0 else test).append(n)
                flip += 1
    state = {
        "current_round": 0, "reps": 1, "split_seed": seed,
        "goal": {"target": TARGET, "direction": "higher", "hold": [GUARD]},
        "approve_each_round": False,
        "best": {"round": 0, "variant": "baseline"},
        "train_ids": sorted(train), "test_ids": sorted(test),
        "harness_paths": HARNESS, "harness_sha": digest(),
        "metrics": [
            {"id": "f3_trim", "kind": "float", "scale": 1, "label": "F 3s trim"},
            {"id": "f05_trim", "kind": "float", "scale": 1, "label": "F 0.5s trim"},
            {"id": "f3", "kind": "float", "scale": 1, "label": "F 3s"},
            {"id": "f05", "kind": "float", "scale": 1, "label": "F 0.5s"},
        ],
        "perf_fields": [{"id": "latency_s", "label": "Time", "unit": "s"}],
    }
    FLOW.mkdir(parents=True, exist_ok=True)
    (FLOW / "_state.json").write_text(json.dumps(state, indent=1))
    print(f"split drawn: {len(train)} train, {len(test)} test")


def paired(a: dict, b: dict, ids: list[str], key: str) -> tuple[float, float]:
    d = np.array([a[i]["grade"][key] - b[i]["grade"][key] for i in ids]) * 100
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--flow", default="song-cuts")
    ap.add_argument("--tracks", default="hx_tracks.json")
    ap.add_argument("--prev", default="song-cuts-97", help="flow whose split assignments are carried over on --init")
    args = ap.parse_args()
    global FLOW
    FLOW = FLOW.parent / args.flow
    if args.init:
        init(args.seed, args.tracks, args.prev)
        return

    state = json.loads((FLOW / "_state.json").read_text())
    if state["harness_sha"] != digest():
        print("WARNING: a harness file changed since the split was drawn; scores may not be comparable")
    train, test = state["train_ids"], state["test_ids"]
    variants = ["baseline"] + sorted((p.name for p in FLOW.glob("v*") if (p / "results.jsonl").exists()), key=lambda s: int(s[1:]))
    ref = rows(args.ref)
    print(f"target {TARGET} (0-100), {len(train)} train / {len(test)} test songs, deltas paired versus {args.ref}\n")
    print("| variant | change | test | train | test delta | train delta | half-second (test) | cuts per song |")
    print("|---|---|---|---|---|---|---|---|")
    for v in variants:
        r = rows(v)
        desc = json.loads((FLOW / v / "summary.json").read_text()).get("description", "")
        mean = lambda ids, k: np.mean([r[i]["grade"][k] for i in ids]) * 100
        cell = ["", ""]
        if v != args.ref:
            for j, ids in enumerate((test, train)):
                m, hw = paired(r, ref, ids, TARGET)
                cell[j] = f"{m:+.1f} ±{hw:.1f}" + ("" if abs(m) > hw else " (within noise)")
        cuts = np.mean([r[i]["meta"]["n_cuts"] for i in train + test])
        print(f"| {v} | {desc} | {mean(test, TARGET):.1f} | {mean(train, TARGET):.1f} | {cell[0]} | {cell[1]} | {mean(test, GUARD):.1f} | {cuts:.1f} |")


if __name__ == "__main__":
    main()
