"""Build the candidate list for the demo's song catalogue by a fixed rule, before any output is looked at.

Usage: uv run scripts/catalogue_list.py [--first 2018] [--last 2025] [--top 40]

The rule: the top N songs of each Billboard Year-End Hot 100 chart in the range, read from Wikipedia's
"Billboard Year-End Hot 100 singles of <year>" pages. A song that appears in two years is kept once.
Songs in the Harmonix Set are left out, because the namer was trained on Harmonix labels.
Writes data/catalogue_candidates.json: rank, year, title, artist (as credited), and the first artist.
"""
import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = "https://en.wikipedia.org/w/index.php?title=Billboard_Year-End_Hot_100_singles_of_{}&action=raw"


def plain(wikitext: str) -> str:
    """Wiki markup to text: [[a|b]] -> b, [[a]] -> a, templates and refs removed."""
    t = re.sub(r"<!--.*?-->|<ref[^>]*?/>|<ref.*?</ref>", "", wikitext, flags=re.S)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    return re.sub(r"\s+", " ", t.replace("''", "")).strip()


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", s.lower()))


def chart(year: int) -> list[dict]:
    req = urllib.request.Request(PAGE.format(year), headers={"User-Agent": "catalogue-builder/0.1"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode()
    rows, carry, left = [], "", 0  # carry: an artist cell that spans the next rows (rowspan)
    for block in raw.split("\n|-")[1:]:
        cells = []
        for line in block.split("\n"):
            if line.startswith("|") and not line.startswith(("|+", "|}")):
                cells += [c.strip() for c in line[1:].split("||")]
        if len(cells) < 2:
            continue
        num = re.sub(r'^scope="row"\s*\|', "", cells[0]).strip()
        if not num.isdigit():
            continue
        title = plain(cells[1]).strip('"“” ')
        if len(cells) > 2:
            span = re.match(r'rowspan="?(\d+)"?\s*\|', cells[2])
            artist = plain(re.sub(r'^rowspan="?\d+"?\s*\|', "", cells[2]))
            if span:
                carry, left = artist, int(span.group(1)) - 1
        elif left > 0:
            artist, left = carry, left - 1
        else:
            continue
        first = re.split(r"\s+(?:featuring|feat\.|and|with|&|x)\s+|,\s*", artist)[0].strip()
        rows.append({"rank": int(num), "year": year, "title": title, "artist": artist, "first_artist": first})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2018)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    harmonix = {key(t["title"]) + "|" + key(t["artist"]) for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text())}
    seen, out, dropped = set(), [], []
    for year in range(args.first, args.last + 1):
        rows = [r for r in chart(year) if r["rank"] <= args.top]
        print(f"{year}: {len(rows)} rows")
        for r in rows:
            k = key(r["title"]) + "|" + key(r["first_artist"])
            if k in seen:
                continue
            seen.add(k)
            (dropped if k in harmonix else out).append(r)
    (ROOT / "data" / "catalogue_candidates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(out)} candidates; {len(dropped)} left out as Harmonix songs: {[d['title'] for d in dropped]}")


if __name__ == "__main__":
    main()
