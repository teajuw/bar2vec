"""Score section names end to end: bar2vec's own cuts (cuts.json), then a namer on those cuts,
against the Harmonix annotations. A measurement, not a round of either climb: nothing is tuned here.

Usage: uv run scripts/nm_e2e.py [--split test|train]

Predicted and annotated sections do not line up one to one, so the score is by time: every second of
an annotated, scored section is compared with the name bar2vec gives that second. Macro-F1 over the
eight labels and the share of seconds named right, pooled over songs, ±95% bootstrap over songs.
Rows: each namer (rules = names.json, trained = v7) on annotated edges and on bar2vec's cuts; a ceiling
for the cuts (each cut section given the annotated label that covers most of it); and rules@moved, the
rules' names from annotated edges carried onto the cut sections, which isolates misplaced edges.
The trained namer was fitted on the tuning songs, so its numbers mean something on the held-out split only.
Writes .claude/hillclimb/section-names-e2e/<split>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import nm_run
from cuts import find_cuts, geometry
from names import LABELS, name_sections
from nm_report import boot

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".claude" / "hillclimb" / "section-names-e2e"
IDX = {label: i for i, label in enumerate(LABELS)}


def overlap_seconds(ref_b, truth, est_b, pred) -> np.ndarray:
    """(labels x labels) seconds: [annotated, named], over annotated sections that are scored."""
    c = np.zeros((len(LABELS), len(LABELS)))
    for a0, a1, t in zip(ref_b[:-1], ref_b[1:], truth):
        if t is None:
            continue
        for e0, e1, p in zip(est_b[:-1], est_b[1:], pred):
            c[IDX[t], IDX[p]] += max(0.0, min(a1, e1) - max(a0, e0))
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    args = ap.parse_args()
    st = json.loads((nm_run.FLOW / "_state.json").read_text())
    ids = st[f"{args.split}_ids"]
    tracks = {t["name"]: t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text())}
    ccfg = json.loads((ROOT / "cuts.json").read_text())
    namers = {"rules": json.loads((ROOT / "names.json").read_text()),
              "trained": json.loads((nm_run.FLOW / "v7" / "names.json").read_text())}
    assert not namers["rules"].get("trained") and namers["trained"].get("trained")
    models = sorted({m for n in namers.values() for m in (n["encoder"], *n.get("extra_models", []))} | set(ccfg["models"]))
    z = {m: np.load(ROOT / "data" / f"hx_{m}.npz") for m in models}
    rules = namers["rules"]
    text = nm_run.text_vectors(rules["encoder"], {**rules["prompts"], **rules.get("extra_prompts", {})})

    rows = ["oracle@annotated", "ceiling@cuts"] + [f"{n}@{b}" for n in namers for b in ("annotated", "cuts")] + ["rules@moved"]
    conf = {r: np.zeros((len(ids), len(LABELS), len(LABELS))) for r in rows}
    n_sec = {"annotated": [], "cuts": []}
    for k, name in enumerate(ids):
        t = tracks[name]
        ref_b, raw = nm_run.reference(name)
        truth = [nm_run.canon(x) for x in raw]
        emb = {m: v["emb"][v["track"] == t["id"]] for m, v in z.items()}
        starts = z[ccfg["models"][0]]["start"][z[ccfg["models"][0]]["track"] == t["id"]]
        end = float(ref_b[-1])
        cuts, _ = find_cuts({m: emb[m] for m in ccfg["models"]}, starts, *geometry(ccfg, starts), ccfg)
        cuts = np.sort(cuts[(cuts > 0) & (cuts < end)])
        bounds = {"annotated": ref_b, "cuts": np.concatenate([[0.0], cuts, [end]])}
        for b in bounds:
            n_sec[b].append(len(bounds[b]) - 1)
        conf["oracle@annotated"][k] = overlap_seconds(ref_b, truth, ref_b, [x or "verse" for x in truth])
        # Ceiling for the cuts: the best single name per cut section, read off the annotation.
        best = []
        for e0, e1 in zip(bounds["cuts"][:-1], bounds["cuts"][1:]):
            sec = np.zeros(len(LABELS))
            for a0, a1, lab in zip(ref_b[:-1], ref_b[1:], truth):
                if lab is not None:
                    sec[IDX[lab]] += max(0.0, min(a1, e1) - max(a0, e0))
            best.append(LABELS[int(sec.argmax())])
        conf["ceiling@cuts"][k] = overlap_seconds(ref_b, truth, bounds["cuts"], best)
        for n, cfg in namers.items():
            bar_starts = z[cfg["encoder"]]["start"][z[cfg["encoder"]]["track"] == t["id"]]
            for b, edges in bounds.items():
                pred, _ = name_sections(edges, bar_starts, emb, text, cfg)
                conf[f"{n}@{b}"][k] = overlap_seconds(ref_b, truth, edges, pred)
                if n == "rules" and b == "annotated":
                    # The names the rules give on annotated edges, carried onto the cut sections (most seconds wins):
                    # what the cuts cost through misplaced edges alone, with the namer undisturbed.
                    moved = []
                    for e0, e1 in zip(bounds["cuts"][:-1], bounds["cuts"][1:]):
                        sec = np.zeros(len(LABELS))
                        for a0, a1, lab in zip(ref_b[:-1], ref_b[1:], pred):
                            sec[IDX[lab]] += max(0.0, min(a1, e1) - max(a0, e0))
                        moved.append(LABELS[int(sec.argmax())])
                    conf["rules@moved"][k] = overlap_seconds(ref_b, truth, bounds["cuts"], moved)

    out = {"split": args.split, "songs": len(ids), "metric": "by time: macro-F1 / share of seconds right, x100, ±95% bootstrap over songs",
           "sections_per_song": {b: round(float(np.mean(v)), 1) for b, v in n_sec.items()}, "rows": {}, "paired": {}, "f1_per_label": {}}
    print(f"{args.split}: {len(ids)} songs, sections per song annotated {out['sections_per_song']['annotated']}, cuts {out['sections_per_song']['cuts']}")
    for r in rows:
        s = boot(conf[r])
        out["rows"][r] = {k: round(v, 1) for k, v in s.items()}
        c = conf[r].sum(0)
        f = 200 * np.diag(c) / np.maximum(c.sum(0) + c.sum(1), 1e-9)
        out["f1_per_label"][r] = {label: round(float(x), 1) for label, x in zip(LABELS, f)}
        print(f"  {r:<18} macro-F1 {s['f1']:5.1f} ±{s['f1_ci']:.1f}   seconds right {s['acc']:5.1f} ±{s['acc_ci']:.1f}")
    for a, b in (("rules@cuts", "rules@annotated"), ("trained@cuts", "trained@annotated"), ("trained@cuts", "rules@cuts"),
                 ("rules@cuts", "ceiling@cuts"), ("rules@moved", "rules@annotated"), ("rules@cuts", "rules@moved")):
        s = boot(conf[a], conf[b])
        out["paired"][f"{a} - {b}"] = {k: round(s[k], 1) for k in ("d_f1", "d_f1_ci", "d_acc", "d_acc_ci")}
        print(f"  {a} - {b}: macro-F1 {s['d_f1']:+.1f} ±{s['d_f1_ci']:.1f}, seconds right {s['d_acc']:+.1f} ±{s['d_acc_ci']:.1f}")
    print("  F1 per label:")
    for r in rows[1:]:
        print(f"    {r:<18} " + "  ".join(f"{label[:6]} {v:4.0f}" for label, v in out["f1_per_label"][r].items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.split}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
