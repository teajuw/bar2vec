"""Build a page that shows every song read back into English, with play buttons to check by ear.

Usage: uv run scripts/make_readout_page.py
Writes readout.html (audio plays from ./audio). Each song opens to a timeline of ten-second
windows, with what each model says stands out there compared with the rest of the song.
"""
import html
from pathlib import Path
from urllib.parse import quote

from make_page import CSS, JS, fmt
from readout import Readout

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("clap_w10_h5", "CLAP"), ("muq_w10_h5", "MuQ-MuLan")]

EXTRA_CSS = """
details{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:0 0 10px}
summary{cursor:pointer}summary .t{display:inline}.who{color:var(--muted);font-size:13px}
.says{font-size:13px;margin:4px 0 0}.says b{font-weight:600;color:var(--muted)}
.row{grid-template-columns:44px 52px 1fr}.row .time{text-align:left}
"""


def tags(pairs: list[tuple[str, float]]) -> str:
    return html.escape("; ".join(p for p, _ in pairs))


def main() -> None:
    e = html.escape
    r = {name: Readout(name) for name, _ in MODELS}
    ix = r[MODELS[0][0]].ix
    parts = []
    for t in ix.tracks:
        rows_by_model = {name: r[name].window_rows(t["id"]) for name, _ in MODELS}
        starts = rows_by_model[MODELS[0][0]][0]
        whole = "".join(
            f'<div class="says"><b>{label}:</b> {tags(r[name].top(r[name].track_row(t["id"]), 4))}</div>' for name, label in MODELS
        )
        rows = []
        for i in range(0, len(starts), 2):
            s = float(starts[i])
            said = "".join(
                f'<div class="says"><b>{label}:</b> {tags(r[name].top(rows_by_model[name][2][i], 3, skip=("genre",)))}</div>'
                for name, label in MODELS
            )
            rows.append(
                f'<div class="row"><button aria-label="Play {e(t["title"])} from {fmt(s)}" data-src="{quote(t["file"])}" '
                f'data-start="{s:.1f}" data-end="{s + ix.window:.1f}" onclick="play(this)">&#9654;</button>'
                f'<div class="time">{fmt(s)}</div><div>{said}</div></div>'
            )
        parts.append(
            f'<details><summary><span class="t">{e(t["title"])}</span> <span class="who">{e(t["artist"])}</span>{whole}</summary>'
            f'{"".join(rows)}</details>'
        )

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Songs read back into English</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>"
        "<h1>Songs read back into English</h1>"
        f'<p class="sub">{len(ix.tracks)} songs · under each title: what the whole song sounds like compared with the library · '
        "open a song for what stands out in each ten seconds compared with the rest of that song</p>"
        f'{"".join(parts)}</main><script>{JS}</script></body></html>'
    )
    (ROOT / "readout.html").write_text(page, encoding="utf-8")
    print(f"wrote readout.html ({len(ix.tracks)} songs)")


if __name__ == "__main__":
    main()
