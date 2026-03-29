# PosterSentry Training

**Training code and data for reproducing the [PosterSentry](https://github.com/fairdataihub/poster-sentry) multimodal scientific poster classifier.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data)

Part of the quality control pipeline for [**posters.science**](https://posters.science), a platform for making scientific conference posters Findable, Accessible, Interoperable, and Reusable (FAIR).

Developed by the [**FAIR Data Innovations Hub**](https://fairdataihub.org/) at the California Medical Innovations Institute (CalMI2).

## Overview

This repository contains everything needed to reproduce the PosterSentry classifier from scratch:

- **Training data**: 3,606 balanced text samples (1,803 poster + 1,803 non-poster) extracted from real PDFs
- **Training script**: Multimodal feature extraction + logistic regression training
- **Corpus classification script**: Batch classify large PDF corpora with parallel processing

## Training Data

### Source

The training data comes from **real scientific documents** — zero synthetic data:

| Class | Count | Source |
|-------|-------|--------|
| **Poster** | 1,803 | Verified scientific posters from Zenodo and Figshare |
| **Non-poster** | 1,803 | Manually confirmed non-posters (papers, proceedings, newsletters, abstract books) |

Sampled from a collection of **30,000+ PDFs** scraped from Zenodo and Figshare as part of the posters.science initiative.

### Format

`data/poster_sentry_train.ndjson` — newline-delimited JSON with `text` and `label` fields:

```json
{"text": "TITLE: Effects of Temperature on Enzyme Kinetics\nAUTHORS: A. Smith...", "label": "poster"}
{"text": "Abstract. We present a novel approach to distributed computing...", "label": "non_poster"}
```

This is the same dataset published on HuggingFace at [fairdataihub/poster-sentry-training-data](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data).

### Note on Label Quality

The poster class is drawn from repository records self-described as posters by their uploaders. When PosterSentry was later applied to the full 30K corpus, approximately 20% of repository-labeled "posters" were reclassified as non-posters, indicating meaningful label noise in the broader source data.

## Training

### Prerequisites

```bash
pip install poster-sentry[dev] tqdm
```

You also need access to the source PDF corpus for full multimodal training (text + visual + structural features). The NDJSON data file contains extracted text only — sufficient for text-only retraining.

### Train the Classifier

```bash
python scripts/train_poster_sentry.py --n-per-class 2000
```

This will:
1. Collect poster and non-poster PDFs from the corpus
2. Extract 542-dimensional feature vectors (512 text + 15 visual + 15 structural)
3. Balance classes and split 85/15 train/test
4. StandardScale features and train LogisticRegression
5. Save the classifier head to `~/.poster_sentry/models/poster_sentry_head.npz`

Training completes in ~40 minutes on CPU (PDF rendering is the bottleneck).

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-per-class` | 5000 | Max samples per class |
| `--test-size` | 0.15 | Test set fraction |
| `--models-dir` | `~/.poster_sentry/models` | Output directory for model head |
| `--export-texts` | None | Export extracted texts as NDJSON |

### Expected Results

```
              precision    recall  f1-score   support

  non_poster     0.8633    0.8856    0.8743       271
      poster     0.8821    0.8593    0.8705       270

    accuracy                         0.8725       541

Top features by |coefficient|:
  size_per_page_kb    coef=+7.6518
  page_count          coef=-5.4937
  file_size_kb        coef=-5.4418
```

## Corpus Classification

Classify a large PDF corpus in batch mode with multiprocessing:

```bash
python scripts/classify_corpus.py \
    --input /path/to/pdf/corpus \
    --output corpus_results/ \
    --workers 40
```

Features:
- Parallel feature extraction with ProcessPoolExecutor (bypasses GIL)
- Single `fitz.open()` per PDF for maximum throughput
- Batch text embedding after extraction
- TSV output + JSON metrics summary

## Feature Architecture

PosterSentry extracts three feature channels per PDF:

| Channel | Dimensions | Features |
|---------|------------|----------|
| **Text** | 512 | model2vec static embeddings (potion-base-32M), L2-normalized |
| **Visual** | 15 | Color stats (RGB mean/std), edge density, FFT spatial complexity, whitespace ratio, color diversity |
| **Structural** | 15 | Page count/dimensions, text block count, font count/size/variance, title score, text density, file size |

Total: **542 dimensions**, classified by StandardScaler + LogisticRegression.

## Related Resources

| Resource | Description |
|----------|-------------|
| [poster-sentry (GitHub)](https://github.com/fairdataihub/poster-sentry) | Installable classifier package |
| [poster-sentry (HuggingFace)](https://huggingface.co/fairdataihub/poster-sentry) | Model weights |
| [poster-sentry-training-data (HuggingFace)](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data) | Dataset on HuggingFace Hub |
| [poster2json](https://github.com/fairdataihub/poster2json) | Poster to structured JSON extraction |
| [posters.science](https://posters.science) | Platform |

## Citation

```bibtex
@dataset{poster_sentry_data_2026,
  title = {PosterSentry Training Data: Scientific Poster Text Corpus},
  author = {O'Neill, James and Soundarajan, Sanjay and Portillo, Dorian and Patel, Bhavesh},
  year = {2026},
  url = {https://github.com/fairdataihub/poster-sentry-training},
  note = {Part of the posters.science initiative at FAIR Data Innovations Hub}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [FAIR Data Innovations Hub](https://fairdataihub.org/) at California Medical Innovations Institute (CalMI2)
- [posters.science](https://posters.science) platform
- Funded by [The Navigation Fund](https://doi.org/10.71707/rk36-9x79) — "Poster Sharing and Discovery Made Easy"
