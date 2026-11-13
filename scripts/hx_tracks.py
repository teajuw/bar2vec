"""Build the list of scoreable Harmonix songs from the alignment results.

Usage: uv run scripts/hx_tracks.py [--min-locked 0.9]

A song is kept when at least `min-locked` of its 10 s chunks agree on one time shift and the whole
annotated recording fits inside our audio at that shift. Annotated time t is then our time
t + offset_s; embedding uses only that span of our audio. Writes data/hx_tracks.json.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-locked", type=float, default=0.9)
    ap.add_argument("--fit-tolerance", type=float, default=3.0, help="seconds of the annotated recording allowed to be missing at either end of our audio")
    args = ap.parse_args()
    meta = {r["File"]: r for r in csv.DictReader((ROOT / "external" / "harmonixset" / "dataset" / "metadata.csv").open())}
    files = {p.stem: p for p in (ROOT / "audio_hx").iterdir()}
    kept, why = [], {"low_locked": 0, "does_not_fit": 0}
    for a in sorted(json.loads((ROOT / "data" / "hx_align.json").read_text()), key=lambda a: a["name"]):
        if "error" in a or a["name"] not in files:
            continue
        ref = float(meta[a["name"]]["Duration"])
        if a["locked"] < args.min_locked:
            why["low_locked"] += 1
        elif a["shift_s"] < -args.fit_tolerance or a["shift_s"] + ref > a["ours_s"] + args.fit_tolerance:
            why["does_not_fit"] += 1
        else:
            m = meta[a["name"]]
            kept.append({"id": len(kept), "name": a["name"], "title": m["Title"], "artist": m["Artist"], "genre": m["Genre"],
                         "file": str(files[a["name"]].relative_to(ROOT)), "ref_duration": ref,
                         "offset_s": a["shift_s"], "locked": a["locked"]})
    (ROOT / "data" / "hx_tracks.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    print(f"kept {len(kept)}; dropped {why}")


if __name__ == "__main__":
    main()
