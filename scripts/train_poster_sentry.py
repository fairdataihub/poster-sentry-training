#!/usr/bin/env python3
# Train PosterSentry on the human-validated survey corpus:
# 3,131 unanimously rated documents + 439 adjudicated documents (2026-08-16).
import csv, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/joneill/Nextcloud/vaults/jmind/calmi2/poster-sentry")
SEED = 42
np.random.seed(SEED)

A = "/home/joneill/pubverse_brett/poster_sentry/paper/analysis/inputs"
ADJ = "/home/joneill/Downloads/postersentry_adjudication_2026-08-16.csv"
ROOT = Path("/home/joneill/Nextcloud/vaults/jmind/calmi2/poster_science/poster-pdf-meta")
OUT = Path("/home/joneill/pubverse_brett/poster_sentry/review/retrain_2026-08-16")
OUT.mkdir(parents=True, exist_ok=True)

PANEL = ["cpc1efk8xl7ogclt0i3zeu3o", "j32isol0qqy827448hqci1dd", "u4v0dvvfdss6i2wfkc7exa12"]

# ---------- 1. human-validated labels ----------
votes = defaultdict(dict)
for r in csv.DictReader(open(f"{A}/evaluations_2026-08-02.csv")):
    if r["userId"] in PANEL:
        votes[r["posterId"]][r["userId"]] = r["userChoice"].replace("-", "_").strip()

adj = {r["posterId"]: r["decision"] for r in csv.DictReader(open(ADJ))}

meta = {r["id"]: r["originalFilename"] for r in csv.DictReader(open(f"{A}/posters_2026-08-02.csv"))}

labels = {}
n_unan = 0
for pid, v in votes.items():
    if pid in adj:
        labels[pid] = adj[pid]
        continue
    c = Counter(x for x in v.values() if x != "unsure")
    assert c and min(c.get("poster", 0), c.get("non_poster", 0)) == 0, f"non-unanimous outside adjudication: {pid}"
    labels[pid] = "poster" if c.get("poster", 0) > 0 else "non_poster"
    n_unan += 1
bal = Counter(labels.values())
print(f"labels: {len(labels)} total | unanimous {n_unan} | adjudicated {len(adj)}")
print(f"class balance: {dict(bal)}")

# ---------- 2. resolve PDF paths ----------
def resolve(fn):
    stem = re.sub(r"^\d+_", "", fn)
    stem = re.sub(r"\.jpg$", "", stem, flags=re.I)
    source = "zenodo" if stem.startswith("zenodo_") else "figshare"
    for base in (ROOT / "downloads" / source, ROOT / "separated_non_posters" / "downloads" / source):
        p = base / f"{stem}.pdf"
        if p.exists():
            return str(p)
    return None

paths, missing = {}, []
for pid in labels:
    p = resolve(meta[pid])
    if p: paths[pid] = p
    else: missing.append(meta[pid])
print(f"paths resolved: {len(paths)} | missing: {len(missing)}")
for m in missing[:8]: print("  missing:", m)

# ---------- 3. feature extraction (parallel) ----------
from poster_sentry.features import VisualFeatureExtractor, PDFStructuralExtractor

def work(item):
    pid, path = item
    import re as _re
    import pdfplumber
    from poster_sentry.features import VisualFeatureExtractor as V, PDFStructuralExtractor as S
    v = V(); s = S()
    try:
        with pdfplumber.open(path) as _pp:
            if len(_pp.pages) == 0:
                return None
            text = _pp.pages[0].extract_text() or ""
        text = _re.sub(r"\s+", " ", text).strip()[:4000]
        img = v.pdf_to_image(path)
        vf = v.extract(img) if img is not None else {n: 0.0 for n in v.FEATURE_NAMES}
        sf = s.extract(path)
        return (pid, text, v.to_vector(vf), s.to_vector(sf))
    except Exception:
        return None

t0 = time.time()
from multiprocessing import Pool
with Pool(12) as pool:
    results = [r for r in pool.imap_unordered(work, list(paths.items()), chunksize=25) if r is not None]
print(f"extracted: {len(results)} of {len(paths)} in {time.time()-t0:.0f}s")

# text-length rule identical to the original training pipeline
kept = [(pid, tx, vv, sv) for pid, tx, vv, sv in results if len(tx) >= 20]
short = len(results) - len(kept)
print(f"dropped for <20 chars of text (matches original pipeline): {short}")

# ---------- 4. near-duplicate removal before the split ----------
def norm(t): return re.sub(r"\W+", "", t.lower())[:300]
seen, dedup, dropped = set(), [], 0
for row in kept:
    k = norm(row[1])
    if k in seen:
        dropped += 1; continue
    seen.add(k); dedup.append(row)
print(f"near-duplicates removed before split: {dropped} | final corpus: {len(dedup)}")

ids = [r[0] for r in dedup]
texts = [r[1] for r in dedup]
y = np.array([1 if labels[i] == "poster" else 0 for i in ids])
print(f"final label balance: poster={int(y.sum())} non_poster={int((1-y).sum())}")

from model2vec import StaticModel
tm = StaticModel.from_pretrained("minishlab/potion-base-32M")
emb = tm.encode(texts)
nrm = np.linalg.norm(emb, axis=1, keepdims=True); nrm = np.where(nrm == 0, 1, nrm)
emb = (emb / nrm).astype("float32")
X = np.concatenate([emb, np.array([r[2] for r in dedup], dtype="float32"),
                    np.array([r[3] for r in dedup], dtype="float32")], axis=1)
print(f"feature matrix: {X.shape}")

# ---------- 5. split, scale, train ----------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
scaler = StandardScaler().fit(X[tr])
Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
print(f"train {len(tr)} | test {len(te)} (test balance: poster={int(y[te].sum())}, non={int((1-y[te]).sum())})")

def fit_eval(Xtr_, ytr_, tag):
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="lbfgs", n_jobs=1, random_state=SEED)
    clf.fit(Xtr_, ytr_)
    pred = clf.predict(Xte)
    print(f"\n===== {tag} =====")
    print(classification_report(y[te], pred, target_names=["non_poster", "poster"], digits=4))
    return clf, accuracy_score(y[te], pred)

# primary: all human-validated data, balanced class weights
clf, acc_full = fit_eval(Xtr, y[tr], "PRIMARY: full corpus + balanced class weights")

# variant: balanced downsampling (original recipe)
pos = np.where(y[tr] == 1)[0]; neg = np.where(y[tr] == 0)[0]
m = min(len(pos), len(neg))
rng = np.random.default_rng(SEED)
keep = np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)])
_, acc_ds = fit_eval(Xtr[keep], y[tr][keep], f"VARIANT: balanced downsample ({m}/class)")

# ---------- 6. old released head on the same held-out set ----------
old = np.load("/home/joneill/Nextcloud/vaults/jmind/calmi2/poster-sentry/models/poster_sentry_head.npz", allow_pickle=True)
oW, ob = old["W"], old["b"]
om, os_ = old["scaler_mean"], old["scaler_scale"]
lab = list(old["labels"]); ip = lab.index("poster")
Xte_old = (X[te] - om) / os_
logits = Xte_old @ oW + ob
old_pred = (logits[:, ip] > logits[:, 1 - ip]).astype(int)
print("\n===== OLD released head on the same held-out set =====")
print(classification_report(y[te], old_pred, target_names=["non_poster", "poster"], digits=4))
acc_old = accuracy_score(y[te], old_pred)

# ---------- 7. top features + save ----------
names = ([f"emb_{i}" for i in range(512)] + list(VisualFeatureExtractor.FEATURE_NAMES)
         + list(PDFStructuralExtractor.FEATURE_NAMES))
coef = clf.coef_[0]
top = np.argsort(np.abs(coef))[-15:][::-1]
print("\nTop 15 features (new PosterSentry):")
for i in top: print(f"  {names[i]:24s} {coef[i]:+.3f}")

W = np.vstack([-clf.coef_[0], clf.coef_[0]]).T.astype("float32")
b = np.array([-clf.intercept_[0], clf.intercept_[0]], dtype="float32")
np.savez(OUT / "poster_sentry_head.npz", W=W, b=b, labels=np.array(["non_poster", "poster"]),
         scaler_mean=scaler.mean_.astype("float32"), scaler_scale=scaler.scale_.astype("float32"))
np.savez(OUT / "features_cache.npz", X=X, y=y, ids=np.array(ids))
json.dump({
    "date": "2026-08-16", "seed": SEED,
    "labels_total": len(labels), "unanimous": n_unan, "adjudicated": len(adj),
    "class_balance": dict(bal), "paths_resolved": len(paths), "extract_failures": len(paths) - len(results),
    "short_text_dropped": short, "near_duplicates_removed": dropped,
    "final_corpus": len(dedup), "final_poster": int(y.sum()), "final_non_poster": int((1 - y).sum()),
    "train": len(tr), "test": len(te),
    "test_accuracy_full_balancedweights": acc_full,
    "test_accuracy_downsampled": acc_ds,
    "test_accuracy_old_released_head": acc_old,
    "top15": [[names[i], float(coef[i])] for i in top],
}, open(OUT / "metrics.json", "w"), indent=1)
print(f"\nSaved head + features + metrics to {OUT}")
print(f"SUMMARY: new(full)={acc_full:.4f}  new(downsampled)={acc_ds:.4f}  old-head={acc_old:.4f}")
