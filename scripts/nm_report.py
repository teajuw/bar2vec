"""Hill-climb bookkeeping for the section-naming climb: status table with noise intervals.

Usage:
  uv run scripts/nm_report.py --init              # copy the song-cuts split, write _state.json (once)
  uv run scripts/nm_report.py [--ref baseline]    # status table; deltas are paired by song versus --ref

Target: macro-F1 over the eight scored labels, pooled over all sections of a split (each label counts
equally). Guardrail: share of sections named right. Both are recomputed from the per-section
(annotated, named) pairs saved in each row. Intervals are 95% bootstrap intervals over songs (2,000
resamples, fixed seed); a delta is "beyond noise" when its paired interval excludes zero.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from names import LABELS

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / ".claude" / "hillclimb" / "section-names"
HARNESS = ["scripts/nm_run.py", "scripts/nm_report.py", "data/hx_tracks.json", "uv.lock"]
IDX = {l: i for i, l in enumerate(LABELS)}


def digest() -> str:
    h = hashlib.sha256()
    for p in HARNESS:
        h.update((ROOT / p).read_bytes())
    return h.hexdigest()[:16]


def confusions(variant: str, ids: list[str], flow: Path = FLOW) -> np.ndarray:
    """(songs, labels, labels) counts: [song, annotated, named]."""
    rows = {r["prompt_id"]: r for r in map(json.loads, (flow / variant / "results.jsonl").read_text().splitlines())}
    c = np.zeros((len(ids), len(LABELS), len(LABELS)))
    for k, i in enumerate(ids):
        for a, p in rows[i]["meta"]["pairs"]:
            c[k, IDX[a], IDX[p]] += 1
    return c


def f1_per_label(c: np.ndarray) -> np.ndarray:
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    return 2 * tp / np.maximum(c.sum(-1) + c.sum(-2), 1e-9)


def scores(c: np.ndarray) -> tuple[float, float]:
    """(macro-F1, share right) x100 for summed counts c (labels x labels)."""
    return float(100 * f1_per_label(c).mean()), float(100 * np.trace(c) / max(c.sum(), 1))


def boot(c: np.ndarray, ref: np.ndarray | None = None, n: int = 2000) -> dict:
    """Point estimates and 95% bootstrap half-widths over songs; with ref, the same for the paired difference."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(c), (n, len(c)))
    w = np.stack([np.bincount(d, minlength=len(c)) for d in draws]).astype(float)  # (n, songs)
    def stat(x):
        cs = np.einsum("ns,sab->nab", w, x)
        return 100 * f1_per_label(cs).mean(-1), 100 * np.trace(cs, axis1=1, axis2=2) / cs.sum((1, 2))
    half = lambda v: float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2)
    f, a = stat(c)
    out = {"f1": scores(c.sum(0))[0], "f1_ci": half(f), "acc": scores(c.sum(0))[1], "acc_ci": half(a)}
    if ref is not None:
        fr, ar = stat(ref)
        out |= {"d_f1": out["f1"] - scores(ref.sum(0))[0], "d_f1_ci": half(f - fr),
                "d_acc": out["acc"] - scores(ref.sum(0))[1], "d_acc_ci": half(a - ar)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--labels", action="store_true", help="also print per-label F1 on train for the newest variant")
    args = ap.parse_args()
    if args.init:
        src = json.loads((FLOW.parent / "song-cuts" / "_state.json").read_text())
        state = {
            "current_round": 0, "reps": 1, "split_seed": src["split_seed"],
            "split_note": "Same random, genre-balanced split as the song-cuts flow.",
            "goal": {"target": "macro_f1 (pooled over sections, computed by nm_report.py)", "direction": "higher", "hold": ["acc"]},
            "approve_each_round": False, "best": {"round": 0, "variant": "baseline"},
            "train_ids": src["train_ids"], "test_ids": src["test_ids"], "final_test_ids": src["final_test_ids"],
            "final_test_note": "Never scored for naming during the climb. Score only the baseline and the final winner, once, at the end.",
            "harness_paths": HARNESS, "harness_sha": digest(),
            "metrics": [{"id": "acc", "kind": "float", "scale": 1, "label": "Sections right"}],
            "perf_fields": [{"id": "latency_s", "label": "Time", "unit": "s"}],
        }
        FLOW.mkdir(parents=True, exist_ok=True)
        (FLOW / "_state.json").write_text(json.dumps(state, indent=1))
        print(f"state written: {len(state['train_ids'])} train, {len(state['test_ids'])} test, {len(state['final_test_ids'])} final (locked)")
        return
    state = json.loads((FLOW / "_state.json").read_text())
    if state["harness_sha"] != digest():
        print("WARNING: harness files changed since the split was approved; scores may not be comparable")
    variants = sorted((p.name for p in FLOW.iterdir() if (p / "results.jsonl").exists() and (p.name == "baseline" or p.name[1:].isdigit())),
                      key=lambda v: -1 if v == "baseline" else int(v[1:]))
    print(f"{'variant':<10}{'test F1':>14}{'train F1':>14}{'test delta':>16}{'train delta':>16}{'test right':>14}")
    for v in variants:
        cell = {}
        for split in ("test", "train"):
            ids = state[f"{split}_ids"]
            cell[split] = boot(confusions(v, ids), confusions(args.ref, ids) if v != args.ref else None)
        te, tr = cell["test"], cell["train"]
        d = lambda x: f"{x['d_f1']:+.1f} ±{x['d_f1_ci']:.1f}" if "d_f1" in x else ""
        print(f"{v:<10}{te['f1']:>8.1f} ±{te['f1_ci']:<4.1f}{tr['f1']:>8.1f} ±{tr['f1_ci']:<4.1f}{d(te):>16}{d(tr):>16}{te['acc']:>8.1f} ±{te['acc_ci']:<4.1f}")
    if args.labels:
        c = confusions(variants[-1], state["train_ids"]).sum(0)
        print(f"\n{variants[-1]} on train, per label: F1 / annotated / named")
        for l, f, a, p in zip(LABELS, f1_per_label(c), c.sum(1), c.sum(0)):
            print(f"  {l:<13}{100 * f:6.1f}{int(a):6d}{int(p):6d}")


if __name__ == "__main__":
    main()
