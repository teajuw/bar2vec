"""Line each downloaded Harmonix song up with the spectrogram of the recording that was annotated.

Usage: uv run scripts/hx_align.py [--workers 6]

The annotations are timestamps on Harmonix's own audio. The dataset publishes a mel spectrogram of
that audio for every song. For each downloaded file this:
  1. computes the same kind of spectrogram from our audio,
  2. slides the reference along ours and takes the single time shift where they agree best,
  3. checks the shift piece by piece: each 10 s chunk of the reference is matched on its own within
     +/-3 s, and counts as "locked" when its own best shift equals the global one (within 0.15 s).

A song whose chunks nearly all lock is the same recording with extra material before or after it,
so annotated time t is our time t + shift. A song with many unlocked chunks is a different edit
(sections added or removed) and cannot be used with one shift.
Writes data/hx_align.json. Nothing is accepted or rejected here; that is decided from the numbers.
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MEL = ROOT / "external" / "melspecs" / "melspecs"
SR, N_FFT, HOP, N_MELS = 22050, 2048, 1024, 80
FRAME = HOP / SR
CHUNK, SEARCH, LOCK = round(10 / FRAME), round(3 / FRAME), 3


def features(power: np.ndarray) -> np.ndarray:
    import librosa

    db = librosa.power_to_db(power, ref=np.max)
    return (db - db.mean(1, keepdims=True)) / (db.std(1, keepdims=True) + 1e-6)


def our_mel(path: Path) -> np.ndarray:
    import librosa

    from embed_clap import decode

    y = decode(path, SR)
    return librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=0)


def align(name_path: tuple[str, str]) -> dict:
    from scipy.signal import correlate

    name, path = name_path
    try:
        ref = features(np.load(MEL / f"{name}-mel.npy"))
        ours = features(our_mel(Path(path)))
    except Exception as e:  # unreadable audio or missing reference
        return {"name": name, "error": str(e)[:120]}
    n, m = ref.shape[1], ours.shape[1]
    xc = np.sum([correlate(ours[b], ref[b], mode="full", method="fft") for b in range(N_MELS)], axis=0)
    shift = int(np.argmax(xc)) - (n - 1)  # ours[:, i + shift] matches ref[:, i]

    locked, sims = [], []
    for i0 in range(0, n - CHUNK + 1, CHUNK):
        r = ref[:, i0 : i0 + CHUNK].ravel()
        r = r / (np.linalg.norm(r) + 1e-9)
        best, best_l = -2.0, None
        for l in range(shift - SEARCH, shift + SEARCH + 1):
            j0 = i0 + l
            if j0 < 0 or j0 + CHUNK > m:
                continue
            o = ours[:, j0 : j0 + CHUNK].ravel()
            s = float(r @ o / (np.linalg.norm(o) + 1e-9))
            if s > best:
                best, best_l = s, l
        if best_l is None:
            locked.append(False)
            sims.append(0.0)
        else:
            locked.append(abs(best_l - shift) <= LOCK)
            sims.append(best)
    return {
        "name": name, "shift_s": round(shift * FRAME, 3), "ref_s": round(n * FRAME, 2), "ours_s": round(m * FRAME, 2),
        "chunks": len(locked), "locked": round(float(np.mean(locked)), 3) if locked else 0.0,
        "sim_median": round(float(np.median(sims)), 3) if sims else 0.0,
        "locked_run": [int(x) for x in locked],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    names = {r["File"] for r in csv.DictReader((ROOT / "external" / "harmonixset" / "dataset" / "metadata.csv").open())}
    files = sorted((p.stem, str(p)) for p in (ROOT / "audio_hx").iterdir() if p.stem in names)
    with Pool(args.workers) as pool:
        out = pool.map(align, files, chunksize=4)
    (ROOT / "data" / "hx_align.json").write_text(json.dumps(out, indent=0))
    ok = [r for r in out if "error" not in r]
    print(f"{len(out)} files, {len(out) - len(ok)} errors, {sum(r['locked'] >= 0.9 for r in ok)} with at least 90% of chunks locked")


if __name__ == "__main__":
    main()
