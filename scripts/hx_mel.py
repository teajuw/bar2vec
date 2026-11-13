"""Window features straight from the Harmonix reference spectrograms, with no pretrained model.

Used as a comparison row: the same cutter, fed plain audio features in place of CLAP embeddings.
The spectrograms are of the annotated recordings themselves and exist for every song, so this row
needs no downloaded audio and no alignment.

Feature for one window: the mean and the spread of each of the 80 log-mel bands over the window,
then each of the 160 numbers standardised across the song's windows. One simple choice, not tuned.

Usage: uv run scripts/hx_mel.py --list     # write data/hxall_tracks.json (every song with a spectrogram)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MEL = ROOT / "external" / "melspecs" / "melspecs"
FRAME = 1024 / 22050


def build(tracks: list[dict], window: float, hop: float, out: Path) -> None:
    import librosa

    embs, track_ids, start_s = [], [], []
    for t in tracks:
        db = librosa.power_to_db(np.load(MEL / f"{t['name']}-mel.npy"), ref=np.max)
        dur = db.shape[1] * FRAME
        starts = list(np.arange(0, max(dur - window, 0) + 1e-9, hop))
        if starts[-1] + window < dur - 1e-6:
            starts.append(dur - window)  # final window flush with the end, as for audio
        feats = []
        for s in starts:
            a, b = int(round(s / FRAME)), int(round((s + window) / FRAME))
            seg = db[:, a:max(b, a + 1)]
            feats.append(np.concatenate([seg.mean(1), seg.std(1)]))
        f = np.array(feats, dtype=np.float32)
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        embs.append(f)
        track_ids += [t["id"]] * len(starts)
        start_s += [float(s) for s in starts]
    np.savez(out, emb=np.concatenate(embs), track=np.array(track_ids, dtype=np.int32),
             start=np.array(start_s, dtype=np.float32), window=np.float32(window))
    print(f"{len(tracks)} tracks, {len(start_s)} windows -> {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        seg = ROOT / "external" / "harmonixset" / "dataset" / "segments"
        rows = []
        for m in csv.DictReader((ROOT / "external" / "harmonixset" / "dataset" / "metadata.csv").open()):
            if (MEL / f"{m['File']}-mel.npy").exists() and (seg / f"{m['File']}.txt").exists():
                rows.append({"id": len(rows), "name": m["File"], "title": m["Title"], "artist": m["Artist"],
                             "genre": m["Genre"], "ref_duration": float(m["Duration"])})
        (ROOT / "data" / "hxall_tracks.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"{len(rows)} songs with a spectrogram and annotations")


if __name__ == "__main__":
    main()
