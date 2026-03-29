#!/usr/bin/env python3
"""
Classify the full poster corpus (30K+ PDFs) using PosterSentry.

Optimized for maximum throughput:
  - Single fitz.open() per PDF (extracts text, visual, structural in one pass)
  - ProcessPoolExecutor for true parallel CPU work (bypasses GIL)
  - Batch text embedding after feature extraction

Usage:
    python scripts/classify_corpus.py \
        --input /storage/poster-pdf-meta_downloads \
        --output corpus_results/ \
        --workers 40
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

# ── Feature names (duplicated here to avoid import in worker) ────

VISUAL_NAMES = [
    "img_width", "img_height", "img_aspect_ratio",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "local_contrast", "color_diversity", "edge_density",
    "spatial_complexity", "white_space_ratio", "high_contrast_ratio",
]

STRUCTURAL_NAMES = [
    "page_count", "page_width_pt", "page_height_pt", "page_aspect_ratio",
    "page_area_sqin", "is_landscape", "text_block_count", "font_count",
    "avg_font_size", "font_size_variance", "title_score", "text_density",
    "line_count", "file_size_kb", "size_per_page_kb",
]


def extract_all_features(pdf_path: str) -> dict:
    """Extract text + visual + structural features from a PDF in ONE open.

    Designed to run in a worker process — no shared state, no imports at
    module level that would break pickling.
    """
    import fitz
    from PIL import Image as PILImage

    result = {
        "path": pdf_path,
        "text": "",
        "visual": np.zeros(15, dtype="float32"),
        "structural": np.zeros(15, dtype="float32"),
        "error": None,
    }

    try:
        path_obj = Path(pdf_path)
        doc = fitz.open(pdf_path)

        if len(doc) == 0:
            doc.close()
            return result

        page = doc[0]
        rect = page.rect

        # ── Text ────────────────────────────────────────────
        text = page.get_text()
        text = re.sub(r"\s+", " ", text).strip()[:4000]
        result["text"] = text

        # ── Structural features ─────────────────────────────
        sf = np.zeros(15, dtype="float32")
        sf[0] = float(len(doc))                              # page_count
        sf[1] = rect.width                                   # page_width_pt
        sf[2] = rect.height                                  # page_height_pt
        sf[3] = rect.width / rect.height if rect.height > 0 else 0  # page_aspect_ratio
        sf[4] = (rect.width / 72.0) * (rect.height / 72.0)  # page_area_sqin
        sf[5] = float(rect.width > rect.height)              # is_landscape

        sf[13] = path_obj.stat().st_size / 1024.0            # file_size_kb
        sf[14] = sf[13] / max(len(doc), 1)                   # size_per_page_kb

        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b.get("type") == 0]
        sf[6] = float(len(text_blocks))                      # text_block_count

        if text_blocks:
            heights = [b["bbox"][3] - b["bbox"][1] for b in text_blocks]
            widths = [b["bbox"][2] - b["bbox"][0] for b in text_blocks]
            total_area = sum(h * w for h, w in zip(heights, widths))
            page_area = rect.width * rect.height
            sf[11] = total_area / page_area if page_area > 0 else 0  # text_density

        fonts = set()
        font_sizes = []
        line_count = 0
        for block in text_blocks:
            for line in block.get("lines", []):
                line_count += 1
                for span in line.get("spans", []):
                    fonts.add(span.get("font", ""))
                    sz = span.get("size", 0)
                    if sz > 0:
                        font_sizes.append(sz)

        sf[7] = float(len(fonts))                            # font_count
        sf[12] = float(line_count)                           # line_count
        if font_sizes:
            sf[8] = float(np.mean(font_sizes))               # avg_font_size
            sf[9] = float(np.var(font_sizes)) if len(font_sizes) > 1 else 0  # font_size_variance
            sf[10] = max(font_sizes) / (np.mean(font_sizes) + 1.0)  # title_score

        result["structural"] = sf

        # ── Visual features (render page → image) ───────────
        vf = np.zeros(15, dtype="float32")
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))  # 72 DPI
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            elif pix.n == 1:
                img = np.stack([img[:, :, 0]] * 3, axis=-1)

            h, w = img.shape[:2]
            vf[0] = float(w)                                 # img_width
            vf[1] = float(h)                                 # img_height
            vf[2] = w / h if h > 0 else 0                   # img_aspect_ratio

            target = (256, 256)
            pil = PILImage.fromarray(img).resize(target, PILImage.Resampling.BILINEAR)
            resized = np.array(pil)

            for i, ch_idx in enumerate(range(3)):
                vf[3 + i] = float(np.mean(resized[:, :, ch_idx]))  # mean_r/g/b
                vf[6 + i] = float(np.std(resized[:, :, ch_idx]))   # std_r/g/b

            gray = np.mean(resized, axis=2)
            vf[9] = float(np.std(gray))                      # local_contrast

            # Color diversity
            small = np.array(pil.resize((32, 32)))
            quantized = (small // 32).astype(np.uint8)
            unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
            vf[10] = unique_colors / 512.0                   # color_diversity

            # Edge density
            gy = np.abs(np.diff(gray, axis=0))
            gx = np.abs(np.diff(gray, axis=1))
            vf[11] = float(np.mean(gy) + np.mean(gx)) / 255.0  # edge_density

            # Spatial complexity (FFT)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            mag = np.abs(fft_shift)
            ch, cw = mag.shape[0] // 2, mag.shape[1] // 2
            radius = min(mag.shape) // 4
            y, x = np.ogrid[:mag.shape[0], :mag.shape[1]]
            center_mask = ((y - ch) ** 2 + (x - cw) ** 2) <= radius ** 2
            total_e = np.sum(mag ** 2)
            low_e = np.sum(mag[center_mask] ** 2)
            vf[12] = 1.0 - (low_e / total_e) if total_e > 0 else 0  # spatial_complexity

            # White space
            white_px = np.sum(np.all(resized > 240, axis=2))
            vf[13] = white_px / (target[0] * target[1])     # white_space_ratio

            # High contrast
            vf[14] = float(np.sum(gray < 50) + np.sum(gray > 240)) / gray.size  # high_contrast_ratio
        except Exception:
            pass

        result["visual"] = vf
        doc.close()

    except Exception as e:
        result["error"] = str(e)

    return result


def collect_pdfs(input_dir: Path) -> list[tuple[str, str]]:
    pdfs = []
    for source_dir in sorted(input_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for pdf in sorted(source_dir.iterdir()):
            if pdf.suffix.lower() == ".pdf":
                pdfs.append((str(pdf), source))
    return pdfs


def main():
    parser = argparse.ArgumentParser(description="Classify poster corpus with PosterSentry")
    parser.add_argument("--input", "-i", required=True,
                        help="Directory with source-organized PDFs (e.g., zenodo/, figshare/ subdirs)")
    parser.add_argument("--output", "-o", default=str(SCRIPT_DIR / "corpus_results"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {input_dir} for PDFs...")
    pdf_entries = collect_pdfs(input_dir)
    logger.info(f"Found {len(pdf_entries):,} PDFs")

    if not pdf_entries:
        logger.error("No PDFs found!")
        return

    # Initialize PosterSentry (main process only — for text model + head)
    from poster_sentry import PosterSentry

    sentry = PosterSentry(
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )
    sentry.initialize()

    tsv_path = output_dir / "corpus_classification.tsv"
    all_results = []
    errors = []

    t0 = time.time()
    n_batches = (len(pdf_entries) + args.batch_size - 1) // args.batch_size

    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "filename", "source", "is_poster", "confidence"])

        for batch_idx in tqdm(range(n_batches), desc="Batches", unit="batch"):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(pdf_entries))
            batch = pdf_entries[start:end]

            batch_paths = [p for p, _ in batch]
            batch_sources = [s for _, s in batch]

            # ── Parallel feature extraction (40 processes) ──
            features = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                features = list(pool.map(extract_all_features, batch_paths))

            # ── Separate valid / failed ─────────────────────
            valid_idx = []
            valid_texts = []
            valid_visual = []
            valid_struct = []
            batch_results = [None] * len(features)

            for i, feat in enumerate(features):
                if feat["error"] is not None:
                    batch_results[i] = {
                        "path": feat["path"],
                        "is_poster": False,
                        "confidence": 0.0,
                        "error": feat["error"],
                    }
                else:
                    valid_idx.append(i)
                    valid_texts.append(feat["text"])
                    valid_visual.append(feat["visual"])
                    valid_struct.append(feat["structural"])

            # ── Batch embed + classify ──────────────────────
            if valid_texts:
                text_embs = sentry.embed_texts(valid_texts)
                visual_arr = np.array(valid_visual, dtype="float32")
                struct_arr = np.array(valid_struct, dtype="float32")
                X = np.concatenate([text_embs, visual_arr, struct_arr], axis=1)

                if sentry.scaler_mean is not None and sentry.scaler_scale is not None:
                    X = (X - sentry.scaler_mean) / np.where(
                        sentry.scaler_scale == 0, 1, sentry.scaler_scale
                    )

                logits = X @ sentry.W + sentry.b
                e = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs = e / e.sum(axis=-1, keepdims=True)

                for j, idx in enumerate(valid_idx):
                    poster_prob = float(probs[j, 1])
                    batch_results[idx] = {
                        "path": features[idx]["path"],
                        "is_poster": poster_prob > 0.5,
                        "confidence": round(poster_prob, 4),
                    }

            # ── Write batch results ─────────────────────────
            for r, src in zip(batch_results, batch_sources):
                writer.writerow([
                    r["path"],
                    Path(r["path"]).name,
                    src,
                    r["is_poster"],
                    f"{r['confidence']:.4f}",
                ])
                all_results.append((r, src))
                if "error" in r:
                    errors.append(r)

        f.flush()

    elapsed = time.time() - t0
    logger.info(f"Classification complete in {elapsed/60:.1f} min ({elapsed/len(pdf_entries):.3f}s/pdf)")

    # Generate metrics
    metrics = _compute_metrics(all_results, elapsed)
    metrics_path = output_dir / "corpus_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if errors:
        errors_path = output_dir / "corpus_errors.json"
        with open(errors_path, "w") as f:
            json.dump(
                [{"path": e["path"], "error": e.get("error", "")} for e in errors],
                f, indent=2,
            )
        logger.info(f"Errors: {len(errors)} (saved to {errors_path})")

    logger.info("=" * 60)
    logger.info("CORPUS CLASSIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total PDFs:      {metrics['total']:,}")
    logger.info(f"  Posters:         {metrics['poster_count']:,} ({metrics['poster_pct']:.1f}%)")
    logger.info(f"  Non-posters:     {metrics['non_poster_count']:,} ({metrics['non_poster_pct']:.1f}%)")
    logger.info(f"  Errors:          {metrics['error_count']:,}")
    logger.info(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
    logger.info(f"  Time:            {elapsed/60:.1f} min")
    logger.info(f"  Results:         {tsv_path}")
    logger.info(f"  Metrics:         {metrics_path}")

    for src in ["zenodo", "figshare"]:
        if src in metrics["by_source"]:
            s = metrics["by_source"][src]
            logger.info(
                f"  [{src}] {s['poster_count']:,} posters / "
                f"{s['total']:,} total ({s['poster_pct']:.1f}%)"
            )


def _compute_metrics(all_results, elapsed):
    total = len(all_results)
    confidences = []
    poster_confs = []
    non_poster_confs = []
    poster_count = 0
    error_count = 0
    by_source = {}

    for r, src in all_results:
        if "error" in r:
            error_count += 1
            continue
        conf = r["confidence"]
        confidences.append(conf)
        is_poster = r["is_poster"]
        if is_poster:
            poster_count += 1
            poster_confs.append(conf)
        else:
            non_poster_confs.append(conf)
        if src not in by_source:
            by_source[src] = {"total": 0, "poster_count": 0, "confidences": []}
        by_source[src]["total"] += 1
        by_source[src]["confidences"].append(conf)
        if is_poster:
            by_source[src]["poster_count"] += 1

    non_poster_count = total - poster_count - error_count
    confs = np.array(confidences) if confidences else np.array([0.0])

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist_counts, _ = np.histogram(confs, bins=bins)

    borderline = [
        (r["path"], r["confidence"])
        for r, _ in all_results
        if "error" not in r and 0.4 <= r["confidence"] <= 0.6
    ]

    source_summary = {}
    for src, data in by_source.items():
        src_confs = np.array(data["confidences"])
        source_summary[src] = {
            "total": data["total"],
            "poster_count": data["poster_count"],
            "non_poster_count": data["total"] - data["poster_count"],
            "poster_pct": 100.0 * data["poster_count"] / max(data["total"], 1),
            "mean_confidence": float(np.mean(src_confs)),
        }

    return {
        "total": total,
        "poster_count": poster_count,
        "non_poster_count": non_poster_count,
        "error_count": error_count,
        "poster_pct": 100.0 * poster_count / max(total - error_count, 1),
        "non_poster_pct": 100.0 * non_poster_count / max(total - error_count, 1),
        "mean_confidence": float(np.mean(confs)),
        "median_confidence": float(np.median(confs)),
        "std_confidence": float(np.std(confs)),
        "confidence_histogram": {
            "bins": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
            "counts": hist_counts.tolist(),
        },
        "poster_mean_confidence": float(np.mean(poster_confs)) if poster_confs else 0.0,
        "non_poster_mean_confidence": float(np.mean(non_poster_confs)) if non_poster_confs else 0.0,
        "high_confidence_count": int(np.sum(confs >= 0.8)),
        "low_confidence_count": int(np.sum(confs <= 0.2)),
        "borderline_count": len(borderline),
        "borderline_samples": [{"path": p, "confidence": c} for p, c in borderline[:50]],
        "by_source": source_summary,
        "elapsed_seconds": elapsed,
        "pdfs_per_second": total / max(elapsed, 1),
    }


if __name__ == "__main__":
    main()
