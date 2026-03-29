#!/usr/bin/env python3
"""
Train PosterSentry on the real posters.science corpus.

Data sources (all real, zero synthetic):
    Positive (poster):
        28K+ verified scientific posters from Zenodo & Figshare
        /home/joneill/Nextcloud/vaults/jmind/calmi2/poster_science/poster-pdf-meta/downloads/

    Negative (non_poster):
        2,036 verified non-posters (multi-page docs, proceedings, abstracts)
        Listed in: poster_classifier/non_posters_20251208_152217.txt

        Plus: single pages extracted from armanc/scientific_papers (real papers)
        Plus: ag_news articles (real junk text, rendered to match)

Usage:
    cd /home/joneill/pubverse_brett/poster_sentry
    pip install -e ".[train]"
    python scripts/train_poster_sentry.py --n-per-class 5000
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────

# Configure these paths to point to your poster corpus.
# Set the POSTER_SCIENCE_BASE environment variable to override.
POSTER_SCIENCE_BASE = Path(
    os.environ.get(
        "POSTER_SCIENCE_BASE",
        str(Path.home() / "poster_science"),
    )
)
DOWNLOADS_DIR = POSTER_SCIENCE_BASE / "poster-pdf-meta" / "downloads"
NON_POSTERS_LIST = (
    POSTER_SCIENCE_BASE
    / "poster_classifier"
    / "non_posters_20251208_152217.txt"
)
CLASSIFICATION_JSON = (
    POSTER_SCIENCE_BASE
    / "poster_classifier"
    / "classification_results_20251208_152217.json"
)


def _fix_path(p: str) -> str:
    """Fix paths from classification JSON if needed.

    The original classification results may reference paths under a
    different mount point.  Set POSTER_SCIENCE_PATH_FROM / _TO to override.
    """
    prefix_from = os.environ.get("POSTER_SCIENCE_PATH_FROM", "/joneill/vaults/")
    prefix_to = os.environ.get("POSTER_SCIENCE_PATH_TO", "/joneill/Nextcloud/vaults/")
    if prefix_from in p and prefix_to not in p:
        return p.replace(prefix_from, prefix_to)
    return p


def collect_poster_paths(max_n: int = 10000) -> List[str]:
    """Collect verified poster PDF paths from the corpus."""
    # Load the classification results to get confirmed poster paths
    if CLASSIFICATION_JSON.exists():
        logger.info(f"Loading classification results from {CLASSIFICATION_JSON}")
        with open(CLASSIFICATION_JSON) as f:
            data = json.load(f)
        poster_entries = data.get("posters", [])
        paths = [_fix_path(e["pdf_path"]) for e in poster_entries if Path(_fix_path(e["pdf_path"])).exists()]
        logger.info(f"  Found {len(paths)} verified poster paths")
    else:
        # Fallback: glob the downloads directory
        logger.info(f"Globbing {DOWNLOADS_DIR} for PDFs...")
        paths = [str(p) for p in DOWNLOADS_DIR.rglob("*.pdf")]
        paths += [str(p) for p in DOWNLOADS_DIR.rglob("*.PDF")]
        logger.info(f"  Found {len(paths)} PDFs")

    random.shuffle(paths)
    return paths[:max_n]


def collect_non_poster_paths(max_n: int = 2000) -> List[str]:
    """Collect verified non-poster PDF paths.

    The non-posters were separated into:
        poster-pdf-meta/separated_non_posters/downloads/{zenodo,figshare}/
    """
    paths = []

    # Primary: glob the separated_non_posters directory
    sep_dir = POSTER_SCIENCE_BASE / "poster-pdf-meta" / "separated_non_posters" / "downloads"
    if sep_dir.exists():
        for pdf in sep_dir.rglob("*.pdf"):
            paths.append(str(pdf))
        for pdf in sep_dir.rglob("*.PDF"):
            paths.append(str(pdf))
        logger.info(f"  Found {len(paths)} non-poster PDFs in {sep_dir}")
    else:
        # Fallback: try the original list with path fixing
        logger.info("  Separated dir not found, trying original list...")
        if NON_POSTERS_LIST.exists():
            with open(NON_POSTERS_LIST) as f:
                for line in f:
                    p = _fix_path(line.strip())
                    if p and Path(p).exists():
                        paths.append(p)
            logger.info(f"  Found {len(paths)} verified non-poster paths from list")

    random.shuffle(paths)
    return paths[:max_n]


def extract_features_from_pdfs(
    pdf_paths: List[str],
    label: int,
    text_model,
    visual_ext,
    structural_ext,
    max_text_chars: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract multimodal features from a list of PDFs.

    Returns (X, y, extracted_texts) where:
        X: (N, 542) feature matrix
        y: (N,) labels
        extracted_texts: list of extracted text strings (for PubGuard reuse)
    """
    from tqdm import tqdm
    import fitz
    import re

    embeddings = []
    visual_vecs = []
    struct_vecs = []
    texts_out = []
    labels = []

    for pdf_path in tqdm(pdf_paths, desc=f"{'poster' if label == 1 else 'non_poster'}"):
        try:
            # Extract text
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                doc.close()
                continue
            text = doc[0].get_text()
            doc.close()
            text = re.sub(r"\s+", " ", text).strip()[:max_text_chars]

            if len(text) < 20:
                continue

            # Visual features
            img = visual_ext.pdf_to_image(pdf_path)
            if img is not None:
                vf = visual_ext.extract(img)
            else:
                vf = {n: 0.0 for n in visual_ext.FEATURE_NAMES}

            # Structural features
            sf = structural_ext.extract(pdf_path)

            texts_out.append(text)
            visual_vecs.append(visual_ext.to_vector(vf))
            struct_vecs.append(structural_ext.to_vector(sf))
            labels.append(label)

        except Exception as e:
            logger.debug(f"Skipping {pdf_path}: {e}")
            continue

    if not texts_out:
        return np.array([]), np.array([]), []

    # Embed all texts at once
    logger.info(f"Embedding {len(texts_out)} texts...")
    emb = text_model.encode(texts_out, show_progress_bar=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb = (emb / norms).astype("float32")

    visual_arr = np.array(visual_vecs, dtype="float32")
    struct_arr = np.array(struct_vecs, dtype="float32")

    X = np.concatenate([emb, visual_arr, struct_arr], axis=1)
    y = np.array(labels)

    return X, y, texts_out


def main():
    parser = argparse.ArgumentParser(description="Train PosterSentry")
    parser.add_argument("--n-per-class", type=int, default=5000,
                        help="Max samples per class (poster/non_poster)")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--export-texts", default=None,
                        help="Export extracted texts as NDJSON for PubGuard retraining")
    args = parser.parse_args()

    from model2vec import StaticModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from poster_sentry.features import VisualFeatureExtractor, PDFStructuralExtractor

    # Models dir
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        models_dir = Path.home() / ".poster_sentry" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding model
    logger.info("Loading model2vec...")
    emb_cache = models_dir / "poster-sentry-embedding"
    if emb_cache.exists():
        text_model = StaticModel.from_pretrained(str(emb_cache))
    else:
        text_model = StaticModel.from_pretrained("minishlab/potion-base-32M")
        emb_cache.parent.mkdir(parents=True, exist_ok=True)
        text_model.save_pretrained(str(emb_cache))

    visual_ext = VisualFeatureExtractor()
    structural_ext = PDFStructuralExtractor()

    # ── Collect data ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Collecting training data...")
    logger.info("=" * 60)

    poster_paths = collect_poster_paths(max_n=args.n_per_class)
    non_poster_paths = collect_non_poster_paths(max_n=args.n_per_class)

    logger.info(f"Poster PDFs to process: {len(poster_paths)}")
    logger.info(f"Non-poster PDFs to process: {len(non_poster_paths)}")

    # ── Extract features ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Extracting features from poster PDFs...")
    logger.info("=" * 60)

    X_pos, y_pos, texts_pos = extract_features_from_pdfs(
        poster_paths, label=1, text_model=text_model,
        visual_ext=visual_ext, structural_ext=structural_ext,
    )

    logger.info(f"Poster features: {X_pos.shape}")

    logger.info("=" * 60)
    logger.info("Extracting features from non-poster PDFs...")
    logger.info("=" * 60)

    X_neg, y_neg, texts_neg = extract_features_from_pdfs(
        non_poster_paths, label=0, text_model=text_model,
        visual_ext=visual_ext, structural_ext=structural_ext,
    )

    logger.info(f"Non-poster features: {X_neg.shape}")

    # ── Balance classes ──────────────────────────────────────────
    min_count = min(len(y_pos), len(y_neg))
    logger.info(f"Balancing: {min_count} samples per class")

    if len(y_pos) > min_count:
        idx = np.random.choice(len(y_pos), min_count, replace=False)
        X_pos = X_pos[idx]
        y_pos = y_pos[idx]
        texts_pos = [texts_pos[i] for i in idx]

    if len(y_neg) > min_count:
        idx = np.random.choice(len(y_neg), min_count, replace=False)
        X_neg = X_neg[idx]
        y_neg = y_neg[idx]
        texts_neg = [texts_neg[i] for i in idx]

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    logger.info(f"Total training data: {X.shape} (poster={sum(y)}, non_poster={len(y)-sum(y)})")

    # ── Export texts for PubGuard ────────────────────────────────
    if args.export_texts:
        export_path = Path(args.export_texts)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            for text in texts_pos:
                f.write(json.dumps({"text": text, "label": "poster"}) + "\n")
            for text in texts_neg:
                f.write(json.dumps({"text": text, "label": "non_poster"}) + "\n")
        logger.info(f"Exported {len(texts_pos) + len(texts_neg)} texts to {export_path}")

    # ── Feature scaling ──────────────────────────────────────────
    # Critical: the 512-d text embedding drowns out the 30 structural/visual
    # features if we don't scale. StandardScaler normalizes each column to
    # zero mean and unit variance, giving structural signals fair weight.
    from sklearn.preprocessing import StandardScaler

    logger.info("=" * 60)
    logger.info("Scaling features (StandardScaler)")
    logger.info("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Log feature variance to confirm structural features are alive
    emb_var = np.mean(np.var(X_scaled[:, :512], axis=0))
    vis_var = np.mean(np.var(X_scaled[:, 512:527], axis=0))
    str_var = np.mean(np.var(X_scaled[:, 527:], axis=0))
    logger.info(f"  Mean variance — text: {emb_var:.3f}  visual: {vis_var:.3f}  structural: {str_var:.3f}")

    # ── Train ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training PosterSentry classifier")
    logger.info("=" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=args.test_size, stratify=y, random_state=SEED,
    )

    logger.info(f"Train: {X_tr.shape[0]:,} | Test: {X_te.shape[0]:,}")
    logger.info(f"Features: {X_tr.shape[1]} (512 text + 15 visual + 15 structural)")

    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="lbfgs", n_jobs=1, random_state=SEED,
    )

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    logger.info(f"Trained in {elapsed:.1f}s")

    y_pred = clf.predict(X_te)
    labels = ["non_poster", "poster"]
    report = classification_report(y_te, y_pred, target_names=labels, digits=4)
    logger.info(f"\n{report}")

    # Show top feature importances
    coef = clf.coef_[0]
    all_names = (
        [f"emb_{i}" for i in range(512)]
        + list(VisualFeatureExtractor.FEATURE_NAMES)
        + list(PDFStructuralExtractor.FEATURE_NAMES)
    )
    top_idx = np.argsort(np.abs(coef))[-15:][::-1]
    logger.info("Top 15 features by |coefficient|:")
    for idx in top_idx:
        logger.info(f"  {all_names[idx]:30s}  coef={coef[idx]:+.4f}")

    # ── Save head as .npz ────────────────────────────────────────
    if clf.coef_.shape[0] == 1:
        W = np.vstack([-clf.coef_[0], clf.coef_[0]]).T.astype("float32")
        b = np.array([-clf.intercept_[0], clf.intercept_[0]], dtype="float32")
    else:
        W = clf.coef_.T.astype("float32")
        b = clf.intercept_.astype("float32")

    head_path = models_dir / "poster_sentry_head.npz"
    np.savez(
        head_path, W=W, b=b, labels=np.array(labels),
        scaler_mean=scaler.mean_.astype("float32"),
        scaler_scale=scaler.scale_.astype("float32"),
    )
    logger.info(f"Saved classifier head + scaler → {head_path}")

    # ── Smoke test ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST")
    logger.info("=" * 60)

    from poster_sentry import PosterSentry

    sentry = PosterSentry(models_dir=models_dir)
    sentry.initialize()

    # Test with some real PDFs
    test_pdfs = poster_paths[:2] + non_poster_paths[:2]
    for p in test_pdfs:
        try:
            result = sentry.classify(p)
            icon = "📋" if result["is_poster"] else "📄"
            print(f"  {icon} {Path(p).name[:60]:60s}  poster={result['is_poster']}  conf={result['confidence']:.3f}")
        except Exception as e:
            print(f"  ⚠️  {Path(p).name[:60]}: {e}")

    logger.info(f"\nDone! Model saved to: {models_dir}")


if __name__ == "__main__":
    main()
