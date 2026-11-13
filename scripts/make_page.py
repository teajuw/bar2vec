"""Build a local page to listen to each query's matched moments.

Usage: uv run scripts/make_page.py [--index clap_w10_h5]
Reads data/eval_<index>.json, writes results_<index>.html. Open it in a browser (audio plays from ./audio).
"""
import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent

CSS = """
:root{--bg:#fbfaf7;--card:#fff;--text:#1f1e1c;--muted:#6b6963;--line:#e3e0d8;--hit:#0f6e56;--hitbg:#e1f5ee;--accent:#534ab7}
@media (prefers-color-scheme: dark){:root{--bg:#1b1a18;--card:#252422;--text:#ece9e2;--muted:#a09d95;--line:#3a3835;--hit:#5dcaa5;--hitbg:#04342c;--accent:#afa9ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:22px;font-weight:600;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 14px}
h2{font-size:16px;font-weight:600;margin:0}.exp{color:var(--muted);font-size:13px;margin:2px 0 10px}
.ranks{font-size:13px;color:var(--muted);margin:0 0 10px}.ranks b{color:var(--text);font-weight:600}
.row{display:grid;grid-template-columns:44px 1fr auto;gap:10px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
button{width:40px;height:40px;border-radius:50%;border:1px solid var(--line);background:var(--bg);color:var(--accent);font-size:15px;cursor:pointer}
button:hover{border-color:var(--accent)}button.on{background:var(--accent);color:var(--bg)}
.t{font-weight:500}.a{color:var(--muted);font-size:13px}.time{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted);text-align:right}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;background:var(--hitbg);color:var(--hit);margin-left:6px;vertical-align:1px}
.base{font-size:13px;color:var(--muted);margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
"""

JS = """
const audio=new Audio();let cur=null,stopAt=0;
audio.addEventListener('timeupdate',()=>{if(stopAt&&audio.currentTime>=stopAt){audio.pause();reset()}});
audio.addEventListener('ended',reset);
function reset(){if(cur){cur.classList.remove('on');cur.textContent='\\u25B6';cur=null}}
function play(b){const same=cur===b;audio.pause();reset();if(same)return;
 const s=+b.dataset.start,e=+b.dataset.end;audio.src=b.dataset.src;
 audio.addEventListener('loadedmetadata',()=>{audio.currentTime=s;audio.play()},{once:true});
 stopAt=e;cur=b;b.classList.add('on');b.textContent='\\u25A0'}
"""


def fmt(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="clap_w10_h5")
    args = ap.parse_args()
    detail = json.loads((ROOT / "data" / f"eval_{args.index}.json").read_text())
    e = html.escape

    parts = []
    for q in detail:
        rows = []
        for m in q["moments"]:
            tag = '<span class="tag">expected track</span>' if m["expected"] else ""
            rows.append(
                f'<div class="row"><button aria-label="Play {e(m["title"])} from {fmt(m["start"])}" '
                f'data-src="{quote(m["file"])}" data-start="{m["start"]:.1f}" data-end="{m["end"]:.1f}" '
                f'onclick="play(this)">&#9654;</button>'
                f'<div><div class="t">{e(m["title"])}{tag}</div><div class="a">{e(m["artist"])}</div></div>'
                f'<div class="time">{fmt(m["start"])}–{fmt(m["end"])}<br>score {m["score"]:.3f}</div></div>'
            )
        r = q["rank"]
        parts.append(
            f'<section><h2>{e(q["query"])}</h2>'
            f'<div class="exp">Expected: {e(", ".join(q["expect"]))}{" · " + e(q["note"]) if q["note"] else ""}</div>'
            f'<div class="ranks">Rank of best expected track — moment search <b>{r["moment"]}</b> · '
            f'whole-track mean <b>{r["mean"]}</b> · one random crop <b>{r["one_crop_median"]:g}</b> (median)</div>'
            f'{"".join(rows)}'
            f'<div class="base">Whole-track mean would return: {e(", ".join(q["mean_top"]))}</div></section>'
        )

    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Moment search results</title><style>{CSS}</style></head><body><main>"
        f"<h1>Moment search results</h1>"
        f'<p class="sub">{len(detail)} test queries · index {e(args.index)} · press play to hear each matched moment</p>'
        f'{"".join(parts)}</main><script>{JS}</script></body></html>'
    )
    out = ROOT / f"results_{args.index}.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
