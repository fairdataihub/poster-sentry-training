#!/usr/bin/env python3
# Re-classify the full corpus with the trained PosterSentry head.
# Checkpointed: processes in batches, resumes from existing parts.
import csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/joneill/Nextcloud/vaults/jmind/calmi2/poster-sentry")
CT = "/home/joneill/pubverse_brett/poster_sentry/corpus_results/corpus_classification.tsv"
NEWROOT = "/home/joneill/Nextcloud/vaults/jmind/calmi2/poster_science/poster-pdf-meta/downloads"
OUT = Path("/home/joneill/pubverse_brett/poster_sentry/review/retrain_2026-08-16")
PARTS = OUT / "corpus_parts"
PARTS.mkdir(parents=True, exist_ok=True)
BATCH = 2500

HEAD = np.load(OUT / "poster_sentry_head.npz", allow_pickle=True)
W, B = HEAD["W"], HEAD["b"]
MEAN, SCALE = HEAD["scaler_mean"], HEAD["scaler_scale"]
IP = list(HEAD["labels"]).index("poster")

rows = list(csv.DictReader(open(CT), delimiter="\t"))
items = [(r["filename"], r["source"],
          r["path"].replace("/storage/poster-pdf-meta_downloads", NEWROOT)) for r in rows]
print(f"corpus rows: {len(items)}", flush=True)

def work(item):
    fn, source, path = item
    import fitz, re as _re
    from poster_sentry.features import VisualFeatureExtractor as V, PDFStructuralExtractor as S
    v = V(); s = S()
    try:
        doc = fitz.open(path)
        if len(doc) == 0:
            doc.close(); return (fn, source, None, None, None)
        text = doc[0].get_text(); doc.close()
        text = _re.sub(r"\s+", " ", text).strip()[:4000]
        img = v.pdf_to_image(path)
        vf = v.extract(img) if img is not None else {n: 0.0 for n in v.FEATURE_NAMES}
        sf = s.extract(path)
        return (fn, source, text, v.to_vector(vf), s.to_vector(sf))
    except Exception:
        return (fn, source, None, None, None)

from model2vec import StaticModel
tm = StaticModel.from_pretrained("minishlab/potion-base-32M")
from multiprocessing import Pool

nb = (len(items) + BATCH - 1) // BATCH
t0 = time.time()
for bi in range(nb):
    part = PARTS / f"part_{bi:03d}.npz"
    if part.exists():
        print(f"batch {bi+1}/{nb}: exists, skipping", flush=True)
        continue
    chunk = items[bi * BATCH:(bi + 1) * BATCH]
    with Pool(10, maxtasksperchild=100) as pool:
        res = list(pool.imap(work, chunk, chunksize=25))
    ok = [(fn, src, tx, vv, sv) for fn, src, tx, vv, sv in res if tx is not None]
    err = [(fn, src) for fn, src, tx, _, _ in res if tx is None]
    if ok:
        texts = [r[2] for r in ok]
        emb = tm.encode(texts)
        nrm = np.linalg.norm(emb, axis=1, keepdims=True); nrm = np.where(nrm == 0, 1, nrm)
        emb = (emb / nrm).astype("float32")
        X = np.concatenate([emb, np.array([r[3] for r in ok], dtype="float32"),
                            np.array([r[4] for r in ok], dtype="float32")], axis=1)
        Xs = (X - MEAN) / SCALE
        logits = Xs @ W + B
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p_poster = (e / e.sum(axis=1, keepdims=True))[:, IP].astype("float32")
        fns = np.array([r[0] for r in ok]); srcs = np.array([r[1] for r in ok])
    else:
        p_poster = np.array([], dtype="float32"); fns = np.array([]); srcs = np.array([])
    np.savez(part, fns=fns, srcs=srcs, p=p_poster,
             err_fns=np.array([e[0] for e in err]), err_srcs=np.array([e[1] for e in err]))
    print(f"batch {bi+1}/{nb}: ok={len(ok)} err={len(err)}  {time.time()-t0:.0f}s", flush=True)

# ---- assemble ----
fns, srcs, ps, errn = [], [], [], 0
for bi in range(nb):
    d = np.load(PARTS / f"part_{bi:03d}.npz", allow_pickle=True)
    fns += list(d["fns"]); srcs += list(d["srcs"]); ps += list(d["p"])
    errn += len(d["err_fns"])
ps = np.array(ps, dtype="float32")
is_poster = ps >= 0.5
with open(OUT / "corpus_classification_new.tsv", "w", newline="") as f:
    wtr = csv.writer(f, delimiter="\t")
    wtr.writerow(["filename", "source", "is_poster", "p_poster"])
    for fn, src, ip_, pp in zip(fns, srcs, is_poster, ps):
        wtr.writerow([fn, src, bool(ip_), f"{pp:.4f}"])
hist, _ = np.histogram(ps, bins=np.linspace(0, 1, 11))
by = {}
for src, ip_ in zip(srcs, is_poster):
    d = by.setdefault(str(src), [0, 0]); d[0] += int(ip_); d[1] += 1
metrics = {
    "total_rows": len(items), "readable": len(ps), "unreadable": errn,
    "poster_count": int(is_poster.sum()), "non_poster_count": int((~is_poster).sum()),
    "poster_pct": float(is_poster.mean() * 100),
    "mean_p_poster": float(ps.mean()),
    "borderline_0.4_0.6": int(((ps >= 0.4) & (ps < 0.6)).sum()),
    "high_conf": int(((ps >= 0.9) | (ps <= 0.1)).sum()),
    "hist_bins_0.1": hist.tolist(),
    "by_source": {k: {"poster": v[0], "n": v[1], "pct": v[0] / v[1] * 100} for k, v in by.items()},
}
json.dump(metrics, open(OUT / "corpus_metrics_new.json", "w"), indent=1)
print(json.dumps({k: metrics[k] for k in ("readable", "poster_count", "poster_pct", "borderline_0.4_0.6")}), flush=True)
print("DONE", flush=True)
