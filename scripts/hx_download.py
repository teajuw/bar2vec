"""Fetch audio for a random sample of the Harmonix Set (section annotations for 912 pop songs).

Usage: uv run scripts/hx_download.py [--n 50] [--seed 0] [--tolerance 2.0]

The dataset ships annotations and a YouTube link per song, not audio. This draws a random sample
of the whole set, downloads each link, and keeps a song only when the audio length is within
`tolerance` seconds of the annotated recording. Writes data/hx_tracks.json and audio_hx/.
A matching length does not prove the audio is the annotated recording or that it starts at the
same instant; a shifted start moves every annotated time.
"""
import argparse
import csv
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HX = ROOT / "external" / "harmonixset" / "dataset"
OUT = ROOT / "audio_hx"


def read(name: str) -> dict[str, dict]:
    return {r["File"]: r for r in csv.DictReader((HX / name).open())}


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def fetch(item: tuple[str, str]) -> tuple[str, Path | None, str]:
    name, url = item
    existing = list(OUT.glob(f"{name}.*"))
    if existing:
        return name, existing[0], ""
    p = subprocess.run(
        ["yt-dlp", "-q", "--no-warnings", "--no-playlist", "-f", "bestaudio[ext=m4a]/bestaudio",
         "-o", str(OUT / f"{name}.%(ext)s"), url],
        capture_output=True, text=True,
    )
    got = list(OUT.glob(f"{name}.*"))
    return name, (got[0] if got else None), p.stderr.strip().splitlines()[-1][:120] if p.stderr.strip() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--slow", type=float, default=0.0, help="seconds to pause between downloads (one at a time)")
    ap.add_argument("--out-json", default="hx_tracks.json", help="write the kept list here (under data/)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    meta, urls, scores = read("metadata.csv"), read("youtube_urls.csv"), read("youtube_alignment_scores.csv")
    names = sorted(n for n in meta if n in urls)
    order = names[:]
    random.Random(args.seed).shuffle(order)  # a larger --n extends the same sample
    sample = order[: args.n]

    items = [(n, urls[n]["URL"]) for n in sample]
    if args.slow:
        # One at a time with a pause; give up once the site refuses several in a row.
        import time

        results, refused = [], 0
        for item in items:
            had = bool(list(OUT.glob(f"{item[0]}.*")))
            r = fetch(item) if refused < 5 else (item[0], None, "skipped: site refusing automated downloads")
            results.append(r)
            refused = refused + 1 if "not a bot" in r[2] else 0 if r[1] is not None and not had else refused
            if not had and refused < 5:
                time.sleep(args.slow)
    else:
        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(fetch, items))

    kept, failed, mismatched = [], [], []
    for name, path, err in results:
        if path is None:
            failed.append((name, err))
            continue
        ref = float(meta[name]["Duration"])
        got = duration(path)
        row = {
            "name": name, "title": meta[name]["Title"], "artist": meta[name]["Artist"], "genre": meta[name]["Genre"],
            "file": str(path.relative_to(ROOT)), "ref_duration": ref, "duration_s": round(got, 2),
            "diff_s": round(got - ref, 2), "alignment_score": float(scores[name]["score"]) if name in scores else None,
        }
        (kept if abs(got - ref) <= args.tolerance else mismatched).append(row)

    for i, r in enumerate(kept):
        r["id"] = i
    (ROOT / "data" / args.out_json).write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    (ROOT / "data" / args.out_json.replace("tracks", "dropped")).write_text(
        json.dumps({"download_failed": failed, "length_mismatch": mismatched}, indent=1, ensure_ascii=False)
    )
    print(f"sampled {args.n} of {len(names)}: kept {len(kept)}, download failed {len(failed)}, "
          f"length differs by more than {args.tolerance:g} s {len(mismatched)}")


if __name__ == "__main__":
    main()
