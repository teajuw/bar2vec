"""Eval runner for the section-naming hill climb: name the annotated sections of every Harmonix case
with the current names.json and score the names against the annotation labels.

Usage: uv run scripts/nm_run.py --variant baseline [--description "..."] [--mode real|oracle|constant|random]
                                [--names names.json] [--out <flow dir>] [--emb-suffix _final]

The namer is given each song's annotated section boundaries and per-bar embeddings, never the labels.
Per case it writes one row to <flow>/<variant>/results.jsonl and a run log to <flow>/<variant>/traces/.
Deterministic, so one rep. Modes other than "real" are harness checks: oracle = the annotation labels
themselves (should score 100), constant = every section "chorus", random = uniform random label (seeded).

Scored labels (see metrics.md): intro, verse, prechorus, chorus, postchorus, bridge, outro, instrumental.
Numbered variants merge (verse2 -> verse); inst, solo, break, transition -> instrumental; anything else
(silence and rare labels) is named but not scored.
"""
import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from names import LABELS, name_sections

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / ".claude" / "hillclimb" / "section-names"
SEG = ROOT / "external" / "harmonixset" / "dataset" / "segments"
MERGE = {"inst": "instrumental", "instrumental": "instrumental", "solo": "instrumental", "break": "instrumental",
         "transition": "instrumental"}


def canon(raw: str) -> str | None:
    """Annotation label -> scored label, or None when the section is not scored."""
    base = re.sub(r"\d+$", "", raw.lower())
    base = MERGE.get(base, base)
    return base if base in LABELS else None


def reference(name: str) -> tuple[np.ndarray, list[str]]:
    """(section edges, raw label per section). The file's last row marks the end of the song."""
    rows = [line.split() for line in (SEG / f"{name}.txt").read_text().splitlines() if line.strip()]
    times, labels = [], []
    for t, label in rows:
        if not times or float(t) > times[-1]:
            times.append(float(t))
            labels.append(label)
    return np.array(times), labels[:-1]


def text_vectors(encoder: str, prompts: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Unit text embeddings per label from the encoder's text side, cached on disk by prompt."""
    assert encoder.startswith("clapbar"), "only CLAP has a text side wired up"
    cache_path = ROOT / "data" / "nm_text_clap.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = sorted({p for ps in prompts.values() for p in ps} - set(cache))
    if missing:
        import contextlib
        import io

        import laion_clap
        import torch

        from embed_clap import CKPT

        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device="mps" if torch.backends.mps.is_available() else "cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            model.load_ckpt(str(CKPT))
        e = np.asarray(model.get_text_embedding(missing + [""] if len(missing) == 1 else missing, use_tensor=False), dtype=np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        cache.update({p: [round(float(x), 6) for x in v] for p, v in zip(missing, e)})
        cache_path.write_text(json.dumps(cache))
    return {label: np.array([cache[p] for p in ps], dtype=np.float32) for label, ps in prompts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--mode", default="real", choices=["real", "oracle", "constant", "random"])
    ap.add_argument("--out", default=str(FLOW))
    ap.add_argument("--names", default=str(ROOT / "names.json"))
    ap.add_argument("--emb-suffix", default="")
    args = ap.parse_args()

    cfg = json.loads(Path(args.names).read_text())
    assert set(cfg["prompts"]) == set(LABELS), "names.json must give prompts for exactly the scored labels"
    st = json.loads((Path(args.out) / "_state.json").read_text())
    keep = set(st["train_ids"]) | set(st["test_ids"])
    tracks = [t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in keep]
    models = sorted({cfg["encoder"], *cfg.get("extra_models", [])})
    z = {m: np.load(ROOT / "data" / f"hx_{m}{args.emb_suffix}.npz") for m in models}
    text = text_vectors(cfg["encoder"], {**cfg["prompts"], **cfg.get("extra_prompts", {})})  # extra_prompts: non-label text the namer may use
    out = Path(args.out) / args.variant
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "names.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copy(ROOT / "scripts" / "names.py", out / "names.py")
    print(f"resolved: {len(tracks)} cases x 1 rep, encoder {cfg['encoder']}, mode {args.mode}")

    rows = []
    for t in tracks:
        t0 = time.time()
        bounds, raw = reference(t["name"])
        truth = [canon(x) for x in raw]
        own = z[cfg["encoder"]]["track"] == t["id"]
        starts = z[cfg["encoder"]]["start"][own]
        assert own.sum() > 4, f"no embeddings for {t['name']}"
        pred, scores = name_sections(bounds, starts, {m: v["emb"][v["track"] == t["id"]] for m, v in z.items()}, text, cfg)
        if args.mode == "oracle":
            pred = [x or "chorus" for x in truth]
        elif args.mode == "constant":
            pred = ["chorus"] * len(truth)
        elif args.mode == "random":
            rng = np.random.default_rng(int(hashlib.sha256(t["name"].encode()).hexdigest()[:8], 16))
            pred = [LABELS[i] for i in rng.integers(0, len(LABELS), len(truth))]
        assert len(pred) == len(truth) and all(p in LABELS for p in pred)
        pairs = [[a, p] for a, p in zip(truth, pred) if a is not None]
        acc = round(float(np.mean([a == p for a, p in pairs])), 4) if pairs else 0.0
        rows.append({
            "prompt_id": t["name"], "rep": 0,
            "prompt": f"{t['title']} — {t['artist']} ({t['genre']}, {len(truth)} sections)",
            "tags": [t["genre"]], "grade": {"acc": acc}, "latency_s": round(time.time() - t0, 4),
            "meta": {"pairs": pairs, "n_scored": len(pairs), "n_sections": len(truth)},
        })
        order = np.argsort(-scores, axis=1)[:, :3]
        trace = [
            {"role": "user", "content": f"Name each section of this song: {rows[-1]['prompt']}\n"
                                        f"section edges (s): {[round(float(b), 1) for b in bounds]}\nsettings: {json.dumps(cfg)}"},
            {"role": "assistant", "content": json.dumps({"names": pred, "top3_scores": [
                [[LABELS[j], round(float(scores[i, j]), 3)] for j in order[i]] for i in range(len(pred))]})},
            {"role": "tool_result", "name": "grader", "content": json.dumps({
                "sections": [{"start": round(float(a), 1), "end": round(float(b), 1), "annotated": r, "scored_as": c,
                              "named": p, "right": (c == p) if c else None}
                             for a, b, r, c, p in zip(bounds[:-1], bounds[1:], raw, truth, pred)],
                "grade": rows[-1]["grade"]})},
        ]
        (out / "traces" / f"{t['name']}_rep0.json").write_text(json.dumps(trace, indent=1))

    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps({"description": args.description or args.variant, "target": "code"}, indent=1))
    print(f"{args.variant}: {len(rows)} rows, mean share of sections right {100 * np.mean([r['grade']['acc'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
