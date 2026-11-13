"""Extra Harmonix songs for TRAINING ONLY: files that did not pass the alignment check for scoring.

Usage: uv run scripts/hx_tracks_extra.py [--min-chunks 6]

hx_tracks.json keeps a song only when at least 90% of its 10 s chunks line up with the annotated
recording and the whole annotated span fits in our audio. That is the right bar for scoring. For
training, a song where only part lines up can still supply labelled bars from the chunks that do.
This lists the unused files with at least --min-chunks locked chunks, leaving out any song that is a
repeat or remix of a song in the held-out or final sets. Each entry carries locked_run so that bars
outside locked chunks get no label. Writes data/hx_tracks_extra.json and a selection file for the
embedding scripts at .claude/hillclimb/section-names-extra/_state.json. Never used for scoring.
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def key(title: str, artist: str) -> tuple[str, str]:
    return (re.sub(r"[^a-z0-9]", "", title.lower().split("(")[0]),
            re.sub(r"[^a-z0-9]", "", artist.lower().split("feat")[0].split(",")[0])[:8])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-chunks", type=int, default=6)
    args = ap.parse_args()
    meta = {r["File"]: r for r in csv.DictReader((ROOT / "external" / "harmonixset" / "dataset" / "metadata.csv").open())}
    files = {p.stem: p for p in (ROOT / "audio_hx").iterdir()}
    used = json.loads((ROOT / "data" / "hx_tracks.json").read_text())
    st = json.loads((ROOT / ".claude" / "hillclimb" / "section-names" / "_state.json").read_text())
    heldout = set(st["test_ids"]) | set(st["final_test_ids"])
    blocked = {key(t["title"], t["artist"]) for t in used if t["name"] in heldout}
    used_names = {t["name"] for t in used}
    kept, why = [], {"too_little_locked": 0, "repeat_of_heldout_song": 0}
    for a in sorted(json.loads((ROOT / "data" / "hx_align.json").read_text()), key=lambda a: a["name"]):
        if "error" in a or a["name"] not in files or a["name"] in used_names:
            continue
        m = meta[a["name"]]
        if sum(a["locked_run"]) < args.min_chunks:
            why["too_little_locked"] += 1
        elif key(m["Title"], m["Artist"]) in blocked:
            why["repeat_of_heldout_song"] += 1
        else:
            kept.append({"id": 10000 + len(kept), "name": a["name"], "title": m["Title"], "artist": m["Artist"], "genre": m["Genre"],
                         "file": str(files[a["name"]].relative_to(ROOT)), "ref_duration": float(m["Duration"]),
                         "offset_s": a["shift_s"], "locked": a["locked"], "locked_run": a["locked_run"]})
    (ROOT / "data" / "hx_tracks_extra.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    flow = ROOT / ".claude" / "hillclimb" / "section-names-extra"
    flow.mkdir(exist_ok=True)
    (flow / "_state.json").write_text(json.dumps({"note": "Selection file only: extra Harmonix songs used for training the bar transformer. Never scored.",
                                                  "train_ids": [t["name"] for t in kept], "test_ids": []}, indent=1))
    mins = sum(sum(t["locked_run"]) for t in kept) / 6
    print(f"kept {len(kept)} extra training songs (about {mins:.0f} min of locked audio); dropped {why}")


if __name__ == "__main__":
    main()
