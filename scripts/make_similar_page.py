"""Build a listening page for "more like this moment" samples, CLAP and MuQ-MuLan side by side.

Usage: uv run scripts/make_similar_page.py [-k 5]
Edit SAMPLES to change the query moments. Writes similar.html (audio plays from ./audio).
"""
import argparse
import html
from pathlib import Path
from urllib.parse import quote

from make_page import CSS, JS, fmt
from search import Index
from similar import find_track, parse_time, similar

ROOT = Path(__file__).resolve().parent.parent

SAMPLES = [
    ("Self Care", "4:30"),
    ("Self Care", "1:00"),
    ("Clarity", "1:00"),
    ("Yellow", "0:10"),
    ("Glimpse of Us", "0:20"),
    ("Zenzenzense", "0:30"),
    ("Nice For What", "0:30"),
    ("Stick Season", "1:00"),
]
MODELS = [("clap_w10_h5", "CLAP"), ("muq_w10_h5", "MuQ-MuLan")]

EXTRA_CSS = """
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media (max-width:640px){.cols{grid-template-columns:1fr}}
.cols h3{font-size:13px;font-weight:600;color:var(--muted);margin:10px 0 4px}
.q{border-top:none;padding-top:10px}.cols .row{grid-template-columns:44px 1fr}
.cols .time{text-align:left}
"""


def button(t: dict, start: float, end: float) -> str:
    return (
        f'<button aria-label="Play {html.escape(t["title"])} from {fmt(start)}" data-src="{quote(t["file"])}" '
        f'data-start="{start:.1f}" data-end="{end:.1f}" onclick="play(this)">&#9654;</button>'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    e = html.escape
    indexes = {name: Index.load(name) for name, _ in MODELS}
    tracks = indexes[MODELS[0][0]].tracks

    parts = []
    for title, at in SAMPLES:
        tid = find_track(indexes[MODELS[0][0]], title)
        results = {name: similar(ix, tid, parse_time(at), args.k) for name, ix in indexes.items()}
        shared = set.intersection(*({m["track"] for m in ms} for _, ms in results.values()))
        start = results[MODELS[0][0]][0]
        window = indexes[MODELS[0][0]].window
        qt = tracks[tid]
        cols = []
        for name, label in MODELS:
            rows = []
            for m in results[name][1]:
                t = tracks[m["track"]]
                tag = '<span class="tag">both models</span>' if m["track"] in shared else ""
                rows.append(
                    f'<div class="row">{button(t, m["start"], m["end"])}'
                    f'<div><div class="t">{e(t["title"])}{tag}</div><div class="a">{e(t["artist"])}</div>'
                    f'<div class="time">{fmt(m["start"])}–{fmt(m["end"])} · similarity {m["score"]:.3f}</div></div></div>'
                )
            cols.append(f'<div><h3>{label}</h3>{"".join(rows)}</div>')
        parts.append(
            f'<section><h2>Sounds like: {e(qt["title"])} at {fmt(start)}</h2>'
            f'<div class="row q">{button(qt, start, start + window)}'
            f'<div><div class="t">{e(qt["title"])}</div><div class="a">{e(qt["artist"])} · the query moment</div></div>'
            f'<div class="time">{fmt(start)}–{fmt(start + window)}</div></div>'
            f'<div class="cols">{"".join(cols)}</div></section>'
        )

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>More like this moment</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>"
        "<h1>More like this moment</h1>"
        f'<p class="sub">{len(SAMPLES)} query moments · no text involved · the query\'s own song is left out · '
        "play the query first, then each match</p>"
        f'{"".join(parts)}</main><script>{JS}</script></body></html>'
    )
    (ROOT / "similar.html").write_text(page, encoding="utf-8")
    print(f"wrote similar.html ({len(SAMPLES)} samples)")


if __name__ == "__main__":
    main()
