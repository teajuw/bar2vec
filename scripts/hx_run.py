"""Eval runner for the song-cut hill climb: run the current cuts.json on every Harmonix case and
record per-case scores and run logs in the hill-climb layout.

Usage: uv run scripts/hx_run.py --variant baseline [--description "..."] [--mode real|oracle|null|even]

Per case it writes one row to <flow>/<variant>/results.jsonl and a run log to
<flow>/<variant>/traces/<id>_rep0.json. Deterministic, so one rep.
Modes other than "real" are harness checks: oracle = the annotated boundaries themselves (should
score 100), null = no cuts (should score 0 trimmed), even = evenly spaced cuts (chance control).
Scores: boundary F (mir_eval) within 3 s and 0.5 s, with and without trimming first/last boundary.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mir_eval
import numpy as np

from cuts import find_cuts, geometry

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / ".claude" / "hillclimb" / "song-cuts"
SEG = ROOT / "external" / "harmonixset" / "dataset" / "segments"


def reference(name: str) -> tuple[np.ndarray, list[str]]:
    rows = [line.split() for line in (SEG / f"{name}.txt").read_text().splitlines() if line.strip()]
    times, labels, seen = [], [], set()
    for t, label in rows:
        if float(t) not in seen:
            seen.add(float(t))
            times.append(float(t))
            labels.append(label)
    return np.array(times), labels


def intervals(bounds: np.ndarray) -> np.ndarray:
    return np.stack([bounds[:-1], bounds[1:]], axis=1)


def grade(ref: np.ndarray, est: np.ndarray) -> dict:
    out = {}
    for key, tol, trim in (("f3_trim", 3.0, True), ("f05_trim", 0.5, True), ("f3", 3.0, False), ("f05", 0.5, False)):
        if trim and (len(ref) < 3 or len(est) < 3):
            out[key] = 0.0
        else:
            out[key] = round(float(mir_eval.segment.detection(intervals(ref), intervals(est), window=tol, trim=trim)[2]), 4)
    return out


def load_embeddings(kind: str, cfg: dict, tracks_file: str = "hx_tracks.json", suffix: str = "") -> dict:
    if "bar_" in kind:  # one feature vector per bar (hx_embed_bars.py, hx_chroma.py)
        return np.load(ROOT / "data" / f"hx_{kind}{suffix}.npz")
    stem = tracks_file.replace("_tracks.json", "")
    path = ROOT / "data" / f"{stem}_{kind}_w{cfg['window']:g}_h{cfg['hop']:g}.npz"
    if not path.exists() and kind == "mel":
        from hx_mel import build

        build(json.loads((ROOT / "data" / tracks_file).read_text()), cfg["window"], cfg["hop"], path)
    elif not path.exists():
        assert stem == "hx", "audio embeddings exist only for the aligned-audio song list"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "hx_embed.py"), "--window", str(cfg["window"]),
                        "--hop", str(cfg["hop"]), "--models", kind], check=True)
    return np.load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--mode", default="real", choices=["real", "oracle", "null", "even"])
    ap.add_argument("--out", default=str(FLOW))
    ap.add_argument("--tracks", default="hx_tracks.json", help="song list under data/")
    ap.add_argument("--models", nargs="+", help="override cuts.json models, e.g. mel (comparison rows only)")
    ap.add_argument("--cuts", default=str(ROOT / "cuts.json"), help="settings file to score (default: the live cuts.json)")
    ap.add_argument("--emb-suffix", default="", help="suffix of the per-bar embedding files, e.g. _final")
    args = ap.parse_args()

    cfg = json.loads(Path(args.cuts).read_text())
    if args.models:
        cfg["models"] = args.models
    tracks = json.loads((ROOT / "data" / args.tracks).read_text())
    state_path = Path(args.out) / "_state.json"
    if state_path.exists():  # score only songs in this flow's split
        st = json.loads(state_path.read_text())
        keep = set(st["train_ids"]) | set(st["test_ids"])
        tracks = [t for t in tracks if t["name"] in keep]
    z = {kind: load_embeddings(kind, cfg, args.tracks, args.emb_suffix) for kind in cfg["models"]}
    out = Path(args.out) / args.variant
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "cuts.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copy(ROOT / "scripts" / "cuts.py", out / "cuts.py")
    print(f"resolved: {len(tracks)} cases x 1 rep, models {cfg['models']}, window {cfg['window']} s, hop {cfg['hop']} s, mode {args.mode}")

    rows = []
    for t in tracks:
        t0 = time.time()
        ref, labels = reference(t["name"])
        end = float(ref[-1])
        first = z[cfg["models"][0]]
        own = first["track"] == t["id"]
        starts = first["start"][own]
        cuts, nov = find_cuts({k: v["emb"][v["track"] == t["id"]] for k, v in z.items()}, starts, *geometry(cfg, starts), cfg)
        if args.mode == "oracle":
            cuts = ref[1:-1]
        elif args.mode == "null":
            cuts = np.array([])
        elif args.mode == "even":
            cuts = np.linspace(0, end, len(cuts) + 2)[1:-1]
        cuts = np.sort(cuts[(cuts > 0) & (cuts < end)])
        est = np.concatenate([[0.0], cuts, [end]])
        g = grade(ref, est)
        rows.append({
            "prompt_id": t["name"], "rep": 0,
            "prompt": f"{t['title']} — {t['artist']} ({t['genre']}, {t['ref_duration']:.0f} s)",
            "tags": [t["genre"]], "grade": g, "latency_s": round(time.time() - t0, 3),
            "meta": {"n_cuts": int(len(cuts)), "n_annotated": int(len(ref) - 2), "offset_s": t.get("offset_s"),
                     "locked": t.get("locked")},
        })
        inner_ref = ref[1:-1]
        trace = [
            {"role": "user", "content": f"Cut this song into sections: {rows[-1]['prompt']}\nsettings: {json.dumps(cfg)}"},
            {"role": "assistant", "content": json.dumps({
                "cuts_s": [round(float(c), 1) for c in cuts],
                "change_score": [[round(float(s), 1), round(float(n), 3)] for s, n in zip(starts, nov)],
            })},
            {"role": "tool_result", "name": "grader", "content": json.dumps({
                "annotated": [[round(float(a), 1), b] for a, b in zip(ref, labels)],
                "cuts_with_nearest_annotation_gap_s": [[round(float(c), 1), round(float(np.min(np.abs(inner_ref - c))), 1) if len(inner_ref) else None] for c in cuts],
                "annotations_with_no_cut_within_3s": [round(float(a), 1) for a in inner_ref if not len(cuts) or np.min(np.abs(cuts - a)) > 3.0],
                "grade": g,
            })},
        ]
        (out / "traces" / f"{t['name']}_rep0.json").write_text(json.dumps(trace, indent=1))

    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps({"description": args.description or args.variant, "target": "code"}, indent=1))
    mean = {k: round(float(np.mean([r["grade"][k] for r in rows])) * 100, 1) for k in rows[0]["grade"]}
    print(f"{args.variant}: {len(rows)} rows, means {mean}")


if __name__ == "__main__":
    main()
