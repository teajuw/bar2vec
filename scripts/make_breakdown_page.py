"""Build a page with a section timeline for every song, one bar per model.

Usage: uv run scripts/make_breakdown_page.py   (after scripts/breakdown.py for each index and scripts/breakdown_bars.py)
Writes breakdown.html (audio plays from ./audio). Click or drag on the bars of a song to scrub; a
playhead shows the position and the current section is highlighted in each bar.
"""
import html
import json
from pathlib import Path
from urllib.parse import quote

from make_page import CSS, fmt

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("clap_w10_h5", "Original cutter, CLAP windows"), ("muq_w10_h5", "Original cutter, MuQ-MuLan windows"),
          ("bar2vec", "bar2vec cutter: one embedding per bar, cuts on bar starts")]
SHORT = {"intro": "intro", "verse": "verse", "prechorus": "pre", "chorus": "chorus", "postchorus": "post", "bridge": "bridge",
         "outro": "outro", "instrumental": "inst"}
COLOURS = ["#AFA9EC", "#5DCAA5", "#F0997B", "#85B7EB", "#EF9F27", "#ED93B1", "#97C459", "#B4B2A9"]

EXTRA_CSS = """
.song{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 12px}
.song h2{display:inline}.who{color:var(--muted);font-size:13px;margin-left:6px}
.lab{font-size:12px;color:var(--muted);margin:10px 0 3px}
.bar{display:flex;gap:2px;height:36px}
.bar button{width:auto;height:36px;min-width:0;border-radius:4px;border:1px solid transparent;color:#1f1e1c;font-size:12px;font-weight:600;padding:0;overflow:hidden;white-space:nowrap}
.bar button:hover{border-color:var(--text)}.bar button.on{outline:2px solid var(--text);outline-offset:-2px;color:#1f1e1c;background-image:linear-gradient(rgba(255,255,255,.28),rgba(255,255,255,.28))}
.ctl{display:flex;align-items:center;gap:10px;margin:8px 0 0}
.pp{width:32px;height:32px;font-size:12px;line-height:1;padding:0}
.ctl .time{text-align:left;min-width:92px}
.bars{position:relative;cursor:pointer;user-select:none;-webkit-user-select:none;touch-action:pan-y}
.head{position:absolute;top:22px;bottom:-4px;left:0;width:2px;margin-left:-1px;background:var(--text);border-radius:1px;pointer-events:none;display:none}
.song.live .head{display:block}
details{margin-top:10px;font-size:13px}summary{cursor:pointer;color:var(--muted)}
.cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}@media (max-width:640px){.cols{grid-template-columns:1fr}}
.seg{padding:4px 0;border-top:1px solid var(--line)}.seg b{font-weight:600}.seg span{color:var(--muted);font-variant-numeric:tabular-nums}
"""

JS = """
const audio=new Audio();let song=null,drag=null;
const fmt=t=>Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
const dur=s=>+s.dataset.dur;
function paint(){if(!song)return;const t=audio.currentTime,d=dur(song);
 song.querySelector('.head').style.left=Math.min(100,t/d*100)+'%';
 song.querySelector('.time').textContent=fmt(t)+' / '+fmt(d);
 song.querySelectorAll('.bar button').forEach(b=>b.classList.toggle('on',t>=+b.dataset.start&&t<+b.dataset.end));
 const pp=song.querySelector('.pp');pp.textContent=audio.paused?'\\u25B6':'\\u275A\\u275A';pp.setAttribute('aria-pressed',!audio.paused)}
function loop(){paint();if(!audio.paused)requestAnimationFrame(loop)}
function clear(s){s.classList.remove('live');s.querySelectorAll('.bar button.on').forEach(b=>b.classList.remove('on'));
 s.querySelector('.pp').textContent='\\u25B6';s.querySelector('.time').textContent='0:00 / '+fmt(dur(s))}
function seek(s,t,play){t=Math.max(0,Math.min(t,dur(s)-0.1));
 if(song!==s){audio.pause();if(song)clear(song);song=s;s.classList.add('live');audio.src=s.dataset.src;
  audio.addEventListener('loadedmetadata',()=>{audio.currentTime=t;if(play)audio.play()},{once:true})}
 else{audio.currentTime=t;if(play&&audio.paused)audio.play()}
 paint()}
function at(bars,e){const r=bars.getBoundingClientRect();return (e.clientX-r.left)/r.width*dur(bars.closest('.song'))}
['play','pause','seeked','timeupdate','ended'].forEach(ev=>audio.addEventListener(ev,()=>{paint();if(ev==='play')loop()}));
document.querySelectorAll('.song').forEach(s=>{
 const bars=s.querySelector('.bars');
 bars.addEventListener('pointerdown',e=>{if(e.button)return;drag=bars;bars.setPointerCapture(e.pointerId);seek(s,at(bars,e),true);e.preventDefault()});
 bars.addEventListener('pointermove',e=>{if(drag===bars)seek(s,at(bars,e),false)});
 ['pointerup','pointercancel'].forEach(ev=>bars.addEventListener(ev,()=>{drag=null}));
 bars.addEventListener('click',e=>{const b=e.target.closest('button');if(b&&e.detail===0)seek(s,+b.dataset.start,true)});
 s.querySelector('.pp').addEventListener('click',()=>{if(song===s){audio.paused?audio.play():audio.pause()}else seek(s,0,true)})});
document.addEventListener('keydown',e=>{if(!song||e.target.closest('summary,a,input'))return;
 if(e.key===' '&&!e.target.closest('button')){e.preventDefault();audio.paused?audio.play():audio.pause()}
 else if(e.key==='ArrowRight'){e.preventDefault();seek(song,audio.currentTime+5,false)}
 else if(e.key==='ArrowLeft'){e.preventDefault();seek(song,audio.currentTime-5,false)}});
"""


def colour(letter: str) -> str:
    return COLOURS[min(ord(letter) - ord("A"), len(COLOURS) - 1)]


def main() -> None:
    e = html.escape
    tracks = json.loads((ROOT / "data" / "tracks.json").read_text())
    data = {name: json.loads((ROOT / "data" / f"breakdown_{name}.json").read_text()) for name, _ in MODELS}

    parts = []
    for t in tracks:
        bars, lists = [], []
        for name, label in MODELS:
            segs = data[name][str(t["id"])]
            buttons = "".join(
                f'<button style="flex:{s["end"] - s["start"]:.1f} 1 0;background:{colour(s["letter"])}" '
                f'title="{fmt(s["start"])}–{fmt(s["end"])}: {e((s["name"] + (" (unsure)" if s.get("unsure") else "") + " — ") if "name" in s else "")}{e("; ".join(s["says"]))}" '
                f'aria-label="Play section {s["letter"]} of {e(t["title"])} from {fmt(s["start"])}" '
                f'data-start="{s["start"]:.2f}" data-end="{s["end"]:.2f}">{SHORT.get(s.get("name"), s["letter"])}{"?" if s.get("unsure") else ""}</button>'
                for s in segs
            )
            bars.append(f'<div class="lab">{label}</div><div class="bar">{buttons}</div>')
            rows = "".join(
                f'<div class="seg"><span>{fmt(s["start"])}–{fmt(s["end"])}</span> <b>{s["letter"]}{(" · " + s["name"] + ("?" if s.get("unsure") else "")) if "name" in s else ""}</b> {e("; ".join(s["says"]))}</div>'
                for s in segs
            )
            lists.append(f'<div><div class="lab">{label}</div>{rows}</div>')
        parts.append(
            f'<div class="song" data-src="{quote(t["file"])}" data-dur="{t["duration_s"]:.2f}"><h2>{e(t["title"])}</h2>'
            f'<span class="who">{e(t["artist"])} · {fmt(t["duration_s"])}</span>'
            f'<div class="ctl"><button class="pp" aria-label="Play or pause {e(t["title"])}" aria-pressed="false">&#9654;</button>'
            f'<span class="time">0:00 / {fmt(t["duration_s"])}</span></div>'
            f'<div class="bars">{"".join(bars)}<div class="head"></div></div><details><summary>Sections in words</summary><div class="cols">{"".join(lists)}</div></details></div>'
        )

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Song breakdowns</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>"
        "<h1>Song breakdowns</h1>"
        f'<p class="sub">{len(tracks)} songs · each bar is the song from start to end, cut where the sound changes · '
        "sections with the same letter sound alike within that bar · letters are not comparable between bars · "
        "on the third bar each section also has a name (intro, verse, pre, chorus, post, bridge, outro, inst) from a classifier trained on 150 annotated songs; "
        "a name ending in ? is one the classifier is unsure of (about half of those are wrong); "
        "colour shows the groups of like-sounding sections the namer worked from · click or drag anywhere on the bars of a song to jump there and keep playing; "
        "the line shows where you are and the section under it is highlighted in each bar · space pauses, left and right arrows skip 5 seconds</p>"
        f'{"".join(parts)}</main><script>{JS}</script></body></html>'
    )
    (ROOT / "breakdown.html").write_text(page, encoding="utf-8")
    print(f"wrote breakdown.html ({len(tracks)} songs)")


if __name__ == "__main__":
    main()
