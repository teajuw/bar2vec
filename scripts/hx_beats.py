"""Beats and downbeats for the Harmonix songs from Beat This!, each song tracked by the fold model that
never trained on it (same fair setup as hx_downbeats.py --fair, which saved only the downbeats).

Usage: uv run scripts/hx_beats.py
Writes data/hx_beats_fold.json: {name: {"beats": [...], "downbeats": [...]}} on the annotated recording's timeline.
"""
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from embed_clap import decode

ROOT = Path(__file__).resolve().parent.parent
SR = 22050


def main() -> None:
    from beat_this.inference import Audio2Beats

    folds = dict(line.split() for line in (ROOT / "data" / "beat_this_harmonix_folds.tsv").read_text().splitlines() if line.strip())
    st = json.loads((ROOT / ".claude" / "hillclimb" / "song-cuts" / "_state.json").read_text())
    keep = set(st["train_ids"]) | set(st["test_ids"])
    tracks = sorted((t for t in json.loads((ROOT / "data" / "hx_tracks.json").read_text()) if t["name"] in keep),
                    key=lambda t: folds[t["name"]])
    trackers, out = {}, {}
    for t in tqdm(tracks):
        ck = f"fold{folds[t['name']]}"
        if ck not in trackers:
            trackers[ck] = Audio2Beats(checkpoint_path=ck, device="mps", dbn=False)
        audio = decode(ROOT / t["file"], SR)
        a = int(round(t["offset_s"] * SR))
        audio = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), audio[max(0, a):]])[: int(t["ref_duration"] * SR)]
        beats, downbeats = trackers[ck](audio, SR)
        out[t["name"]] = {"beats": [round(float(x), 3) for x in beats], "downbeats": [round(float(x), 3) for x in downbeats]}
    (ROOT / "data" / "hx_beats_fold.json").write_text(json.dumps(out))
    old = json.loads((ROOT / "data" / "hx_downbeats_fold.json").read_text())
    same = sum(np.allclose(out[k]["downbeats"], old[k], atol=0.002) if len(out[k]["downbeats"]) == len(old[k]) else False for k in out)
    print(f"{len(out)} songs; downbeats identical to the saved ones in {same}")


if __name__ == "__main__":
    main()
