"""Name each section of a song. This file and names.json are the surface tuned by the section-names
hill climb; scripts/nm_run.py scores them. Nothing here is trained and nothing here sees an annotation label.
"""
import numpy as np

LABELS = ["intro", "verse", "prechorus", "chorus", "postchorus", "bridge", "outro", "instrumental"]


def section_vectors(bounds: np.ndarray, bar_starts: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """One unit vector per section: the mean of the bars that start inside it (nearest bar if none do)."""
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        inside = (bar_starts >= a - 0.05) & (bar_starts < b - 0.05)
        v = emb[inside].mean(0) if inside.any() else emb[np.argmin(np.abs(bar_starts - a))]
        out.append(v / max(np.linalg.norm(v), 1e-9))
    return np.stack(out)


def repeat_groups(vecs: np.ndarray, tau: float) -> np.ndarray:
    """Group sections that sound alike: average-link merging of section vectors (song mean removed)
    until no two groups have mean similarity of at least tau. Returns a group id per section."""
    v = vecs - vecs.mean(0)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    sim = v @ v.T
    groups = [[i] for i in range(len(v))]
    while len(groups) > 1:
        best, pair = -np.inf, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                m = sim[np.ix_(groups[i], groups[j])].mean()
                if m > best:
                    best, pair = m, (i, j)
        if best < tau:
            break
        groups[pair[0]] += groups.pop(pair[1])
    out = np.zeros(len(v), dtype=int)
    for k, members in enumerate(groups):
        out[members] = k
    return out


def by_structure(bounds: np.ndarray, vecs: np.ndarray, st: dict, vocal: np.ndarray | None = None,
                 loud: np.ndarray | None = None) -> list[str]:
    """Names from the song's shape alone: where a section sits and which sections repeat."""
    n = len(vecs)
    dur = np.diff(bounds)
    out = ["verse"] * n
    if st.get("repeat_tau") is not None:
        g = repeat_groups(vecs, st["repeat_tau"])
        count = np.bincount(g)
        repeating = [k for k in range(len(count)) if count[k] >= 2]
        if repeating:
            # The chorus is the group that comes back most often; ties go to the one sitting later in the song.
            # With loud_w, a group's loudness relative to the song's other sections counts too.
            w = st.get("loud_w", 0) if loud is not None else 0
            chorus = max(repeating, key=lambda k: (count[k] + (w * loud[g == k].mean() if w else 0),
                                                   np.mean(np.flatnonzero(g == k))))
            for i in range(n):
                if g[i] == chorus:
                    out[i] = "chorus"
                elif st.get("bridge") and count[g[i]] == 1 and i > n * 0.5:
                    out[i] = "bridge"  # heard once, in the second half
            if st.get("prepost"):
                for k in repeating:
                    if k == chorus:
                        continue
                    mem = np.flatnonzero(g == k)
                    if dur[mem].mean() >= st.get("short_s", 20):
                        continue
                    leads_in = np.mean([i + 1 < n and g[i + 1] == chorus for i in mem])
                    follows = np.mean([i > 0 and g[i - 1] == chorus for i in mem])
                    after_verse = np.mean([i > 0 and out[i - 1] == "verse" and g[i - 1] != k for i in mem])
                    if leads_in >= 0.6 and after_verse >= 0.5:
                        for i in mem:
                            out[i] = "prechorus"
                    elif follows >= 0.6 and leads_in < 0.5:
                        for i in mem:
                            out[i] = "postchorus"
    if st.get("vocal_below") is not None and vocal is not None:
        # A verse or bridge with far less singing than the rest of the song is an instrumental passage.
        for i in range(1, n - 1):
            if vocal[i] < st["vocal_below"] and out[i] in ("verse", "bridge"):
                out[i] = "instrumental"
    if st.get("position"):
        out[0] = "intro"
        if out[-1] != "chorus" or dur[-1] < 12:
            out[-1] = "outro"
    return out


def name_sections(bounds: np.ndarray, bar_starts: np.ndarray, emb_by_model: dict[str, np.ndarray],
                  text: dict[str, np.ndarray], cfg: dict) -> tuple[list[str], np.ndarray]:
    """Return (one label per section, text-match score matrix sections x LABELS).

    bounds: section edges in seconds (n+1 values). text: label -> (prompts, dim) unit vectors from the
    text side of cfg["encoder"]. Without cfg["structure"], a section gets the label whose best prompt
    is closest to its audio. With it, names come from by_structure() on cfg["structure"]["model"] vectors.
    """
    vecs = section_vectors(bounds, bar_starts, emb_by_model[cfg["encoder"]])
    scores = np.stack([(vecs @ text[label].T).max(1) for label in LABELS], axis=1)
    st = cfg.get("structure")
    if not st:
        return [LABELS[i] for i in scores.argmax(1)], scores
    vocal = None
    if "vocals" in text:  # how much each section sounds sung, relative to the song's median section
        d = (vecs @ text["vocals"].T).max(1) - (vecs @ text["no_vocals"].T).max(1)
        vocal = (d - np.median(d)) / (d.std() + 1e-9)
    loud = None
    if st.get("loud_model"):  # mean level of each section's bars in decibels, as a z-score across the song's sections
        level = emb_by_model[st["loud_model"]][:, 0]
        per = np.array([level[(bar_starts >= a - 0.05) & (bar_starts < b - 0.05)].mean()
                        if ((bar_starts >= a - 0.05) & (bar_starts < b - 0.05)).any() else level.mean()
                        for a, b in zip(bounds[:-1], bounds[1:])])
        loud = (per - per.mean()) / (per.std() + 1e-9)
    return by_structure(bounds, section_vectors(bounds, bar_starts, emb_by_model[st["model"]]), st, vocal, loud), scores
