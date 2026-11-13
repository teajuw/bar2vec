"""Download audio for each playlist track, picking the search hit whose duration best matches Spotify's.

Usage: uv run scripts/download.py
Reads data/playlist.json, writes audio/<nn>_<slug>.<ext> and data/downloads.json.
"""
import json
import re
import sys
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
AUDIO.mkdir(exist_ok=True)
OUT = ROOT / "data" / "downloads.json"
MAX_DIFF_OK = 4.0  # seconds; beyond this the match is flagged


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50]


def main() -> None:
    tracks = json.loads((ROOT / "data" / "playlist.json").read_text())
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    seen = set()
    for i, t in enumerate(tracks, 1):
        uri = t["uri"]
        if uri in seen:
            print(f"[{i:02d}] duplicate of an earlier track, skipping: {t['title']}")
            continue
        seen.add(uri)
        if "file" in results.get(uri, {}) and Path(ROOT / results[uri]["file"]).exists():
            continue
        want = t["ms"] / 1000
        query = f"{t['artist']} - {t['title']}"
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
                info = ydl.extract_info(f"ytsearch6:{query}", download=False)
            entries = [e for e in info.get("entries", []) if e and e.get("duration")]
            if not entries:
                raise RuntimeError("no search results")
            best = min(entries, key=lambda e: abs(e["duration"] - want))
            diff = best["duration"] - want
            stem = f"{i:02d}_{slug(t['artist'].split(',')[0])}_{slug(t['title'])}"
            opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "bestaudio[ext=m4a]/bestaudio",
                "outtmpl": str(AUDIO / f"{stem}.%(ext)s"),
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                got = ydl.extract_info(f"https://www.youtube.com/watch?v={best['id']}", download=True)
                path = Path(ydl.prepare_filename(got))
            results[uri] = {
                "index": i,
                "title": t["title"],
                "artist": t["artist"],
                "spotify_s": round(want, 1),
                "video_id": best["id"],
                "video_title": best.get("title"),
                "video_s": best["duration"],
                "diff_s": round(diff, 1),
                "flag": abs(diff) > MAX_DIFF_OK,
                "file": str(path.relative_to(ROOT)),
            }
            mark = "  <-- duration mismatch" if abs(diff) > MAX_DIFF_OK else ""
            print(f"[{i:02d}] {query} -> {best.get('title')} ({diff:+.1f}s){mark}", flush=True)
        except Exception as e:  # keep going; report at the end
            results[uri] = {"index": i, "title": t["title"], "artist": t["artist"], "error": str(e)[:300]}
            print(f"[{i:02d}] FAILED {query}: {str(e)[:200]}", flush=True)
        OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    ok = [r for r in results.values() if "file" in r]
    bad = [r for r in results.values() if "error" in r]
    flagged = [r for r in ok if r["flag"]]
    print(f"\ndone: {len(ok)} downloaded, {len(flagged)} flagged for duration, {len(bad)} failed")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
