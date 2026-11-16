"""Find an official, embeddable YouTube upload for each catalogue candidate.

Usage: uv run scripts/catalogue_match.py [--limit N]

Reads data/catalogue_candidates.json (from catalogue_list.py). For each song it searches YouTube and takes
the first result that passes every check; a song with no passing result is left out of the catalogue.
The checks, fixed before any output was looked at:
  - the upload is the artist's own: a channel YouTube marks as verified whose name contains the first
    artist's name, or the auto-generated "<artist> - Topic" channel;
  - the video's title contains the song's title, and has none of: live, remix, cover, karaoke, slowed,
    sped up, instrumental, acoustic, reaction, parody (unless the song's own title has the word);
  - it is 1.5 to 7 minutes long, allows embedding, is not age-restricted and is not a live stream.
Uploads that are audio only (official audio, lyric video, visualizer, Topic) are preferred over music
videos, which sometimes add an intro or a scene that is not in the song.
The six songs already in article/demo/songs.json keep the uploads they have.
Nothing is downloaded here. Requests are spaced out (--pause seconds between them). When YouTube starts
refusing requests the run stops, and a song whose every lookup errored is not recorded, so the next run
retries it. Results are cached in data/catalogue_match_cache.json, so a rerun continues.
Writes data/catalogue_songs.json (the matched songs) and prints the songs left out and why.
"""
import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAD = ["live", "remix", "cover", "karaoke", "slowed", "sped up", "instrumental", "acoustic", "reaction", "parody", "8d", "mashup", "clean", "tiktok"]
AUDIO = ["official audio", "audio", "lyric", "visualizer", "visualiser"]


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def title_key(s: str) -> str:
    return key(re.sub(r"\(.*?\)|\[.*?\]", "", s))


def judge(info: dict, song: dict) -> tuple[int, str]:
    """(score, reason): score 0 is a rejection; 2 is an audio-only upload, 1 another official upload."""
    vt, channel = info.get("title") or "", info.get("channel") or info.get("uploader") or ""
    topic = channel.endswith(" - Topic")
    artist = key(song["first_artist"])
    if not (topic or (info.get("channel_is_verified") and artist[:8] in key(channel))):
        return 0, f"not the artist's channel ({channel})"
    if topic and artist[:8] not in key(channel) and artist[:8] not in key(" ".join(info.get("artists") or []) + (info.get("artist") or "")):
        return 0, f"another artist's Topic channel ({channel})"
    if title_key(song["title"]) not in key(vt):
        return 0, f"title does not match ({vt})"
    low, own = " " + re.sub(r"[^a-z0-9]+", " ", vt.lower()) + " ", song["title"].lower()
    for w in BAD:
        if f" {w} " in low and w not in own:
            return 0, f"'{w}' in the video title ({vt})"
    dur = info.get("duration") or 0
    if not 90 <= dur <= 420:
        return 0, f"{dur} s long"
    if info.get("playable_in_embed") is False:
        return 0, "embedding is turned off"
    if info.get("age_limit") or info.get("is_live") or info.get("live_status") not in (None, "not_live"):
        return 0, "age-restricted or live"
    return (2 if topic or any(w in vt.lower() for w in AUDIO) else 1), "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--results", type=int, default=6, help="search results looked at per song")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between requests to YouTube")
    args = ap.parse_args()
    import yt_dlp

    cands = json.loads((ROOT / "data" / "catalogue_candidates.json").read_text())[: args.limit or None]
    fixed = {key(s["title"]): s for s in json.loads((ROOT / "article" / "demo" / "songs.json").read_text())}
    cache_path = ROOT / "data" / "catalogue_match_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    flat = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True})
    full = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True})
    out, missed, refused = [], [], 0
    for n, song in enumerate(cands):
        if refused >= 3:
            print("YouTube is refusing requests; stopping here. Run again later to continue.")
            break
        ck = f"{song['title']}|{song['first_artist']}"
        if key(song["title"]) in fixed and key(fixed[key(song["title"])]["artist"]) == key(song["first_artist"]):
            out.append({**fixed[key(song["title"])], "year": song["year"], "rank": song["rank"], "kind": "kept"})
            continue
        if ck not in cache:
            best, reasons, errors = None, [], 0
            time.sleep(args.pause)
            try:
                hits = flat.extract_info(f"ytsearch{args.results}:{song['first_artist']} {song['title']} official audio", download=False)["entries"]
            except Exception as e:  # a search that fails is retried on the next run
                print(f"  search failed for {ck}: {str(e)[:80]}")
                continue
            for h in hits:
                time.sleep(args.pause)
                try:
                    info = full.extract_info(f"https://www.youtube.com/watch?v={h['id']}", download=False)
                except Exception as e:
                    reasons.append(f"{h['id']}: {str(e)[:60]}")
                    errors += 1
                    continue
                score, why = judge(info, song)
                if score == 0:
                    reasons.append(why)
                    continue
                if best is None or score > best["score"]:
                    best = {"score": score, "video": info["id"], "channel": info.get("channel"), "video_title": info.get("title"), "duration": info.get("duration")}
                if score == 2:
                    break
            if best is None and hits and errors == len(hits):  # nothing could be looked at: not a result, try again next run
                refused += 1
                print(f"  every lookup failed for {ck}")
                continue
            refused = 0
            cache[ck] = best or {"score": 0, "reasons": reasons}
            cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n")
        got = cache[ck]
        if got["score"]:
            out.append({"video": got["video"], "title": song["title"], "artist": song["artist"], "year": song["year"], "rank": song["rank"],
                        "kind": "audio" if got["score"] == 2 else "video"})
            print(f"{n + 1:3d} ok   {song['first_artist']} - {song['title']}  <-  {got['channel']}: {got['video_title']} ({got['duration']} s)")
        else:
            missed.append((ck, got["reasons"]))
            print(f"{n + 1:3d} MISS {ck}: {got['reasons'][:2]}")
    if not args.limit and refused < 3:
        (ROOT / "data" / "catalogue_songs.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(out)} matched, {len(missed)} left out of {len(cands)}")


if __name__ == "__main__":
    main()
