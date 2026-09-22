# PosterSentry Training

**Training code and data for reproducing the [PosterSentry](https://github.com/fairdataihub/poster-sentry) multimodal scientific poster classifier.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data)

Part of the quality control pipeline for [**posters.science**](https://posters.science), a platform for making scientific conference posters Findable, Accessible, Interoperable, and Reusable (FAIR).

Developed by the [**FAIR Data Innovations Hub**](https://fairdataihub.org/) at the California Medical Innovations Institute (CalMI2).

## Overview

This repository contains the data and the as-run code behind the released PosterSentry classifier:

- **Training data**: 3,298 documents with human-validated labels (1,651 poster, 1,647 non-poster)
- **Training script**: label construction from the survey and adjudication, multimodal feature extraction, and two-stage (stacked) logistic regression training
- **Corpus classification script**: batch classification of the full 30,205-document corpus with the trained head

Release 1.0.0 supersedes the earlier heuristic-label release, which remains available in the repository history.

## Training Data

### Source and labeling

The training data comes from **real scientific documents** with **human-validated labels**, zero synthetic data. Three reviewers independently classified 3,486 candidate documents that passed license screening at [survey.posters.science](https://survey.posters.science) (inter-rater Krippendorff's alpha 0.79), and the 430 documents without a unanimous panel were settled in a blinded adjudication review. After removing 181 near-duplicates and 7 documents with unavailable PDFs, the remaining 3,298 form the training corpus:

| Class | Count | Label provenance |
|-------|-------|------------------|
| **Poster** | 1,651 | Unanimous panel label or blinded adjudication |
| **Non-poster** | 1,647 | Unanimous panel label or blinded adjudication |

Candidates were drawn from a collection of **30,000+ PDFs** scraped from Zenodo and Figshare as part of the posters.science initiative. When the trained classifier was applied back to that full corpus, it classified 80.5% of repository-labeled "posters" as posters: roughly one in five records labeled as posters is something else.

### Format

`data/poster_sentry_train.ndjson` is newline-delimited JSON, one row per document:

```json
{"id": "fxd4ylwf0byrtj307b5k3kpm", "doi": "10.5281/zenodo.1234567", "source": "zenodo", "text": "TITLE: Effects of Temperature on Enzyme Kinetics ...", "label": "poster", "label_source": "unanimous_panel"}
```

| Field | Description |
|-------|-------------|
| `id` | Survey document identifier |
| `doi` | DOI of the source repository record (present for every row) |
| `source` | `zenodo` or `figshare` |
| `text` | First-page text extracted with pdfplumber, whitespace-normalized, truncated to 4,000 characters |
| `label` | `poster` or `non_poster` (human-validated) |
| `label_source` | `unanimous_panel` (2,949 rows) or `adjudicated` (432 rows) |

This is the same dataset published on HuggingFace at [fairdataihub/poster-sentry-training-data](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data) (release 1.0.0).

## Training

`scripts/train_poster_sentry.py` is the as-run training script. It builds the human-validated labels from the survey votes and adjudication decisions, resolves the source PDFs, extracts the feature channels (512-d text embedding + 15 visual + 15 structural) in parallel, trains the two-stage classifier on a stratified 85/15 split at seed 42 (stage 1 scores the text embedding; its inner 5-fold out-of-fold poster probability becomes the text_score feature of the 31-feature final classifier), and saves the head as a NumPy archive (both stages' weights, biases, and scaler parameters plus the label mapping; about 20 KB).

The script requires the local PDF store harvested with poster-repo-scraper, so it documents the training as run rather than serving as a portable tool; the NDJSON data file contains the extracted text for every training document, sufficient for text-only retraining without the PDFs.

### Expected results (held-out split, n = 508)

```
              precision    recall  f1-score   support

  non_poster     0.926     0.933     0.930       255
      poster     0.932     0.925     0.929       253

    accuracy                         0.9291      508

Top stage-2 features by |coefficient|:
  page_count          coef=-3.87
  size_per_page_kb    coef=+2.49
  line_count          coef=+2.00
  file_size_kb        coef=-1.74
  mean_g              coef=+1.30
```

## PDF backend

Feature extraction uses a selectable backend, defaulting to `pdfplumber`
(pdfplumber for text and structure, pypdfium2 for rendering) — the permissively
licensed stack the released model is trained on. To retrain with the faster but
AGPL-licensed PyMuPDF backend instead, install it and set the environment
variable; no code change is needed:

```bash
pip install poster-sentry[pymupdf]
POSTER_SENTRY_BACKEND=pymupdf python scripts/train_poster_sentry.py --n-per-class 2000
```

The backends extract slightly different features, so a model trained with one
backend should be used for inference with the same backend.

## Corpus Classification

`scripts/classify_corpus.py` classifies the full corpus with the trained head in resumable batches: parallel feature extraction with multiprocessing, checkpointed part files so an interrupted run resumes where it stopped, batch text embedding, and TSV output with a JSON metrics summary.

## Feature Architecture

PosterSentry extracts three feature channels per PDF:

| Channel | Dimensions | Features |
|---------|------------|----------|
| **Text** | 512 | model2vec static embeddings (potion-base-32M), L2-normalized |
| **Visual** | 15 | Color stats (RGB mean/std), edge density, FFT spatial complexity, whitespace ratio, color diversity |
| **Structural** | 15 | Page count/dimensions, text block count, font count/size/variance, title score, text density, file size |

Stage 1 (StandardScaler + LogisticRegression) summarizes the 512-d text embedding into a single text score; the final classifier is a second StandardScaler + LogisticRegression over **31 features** (text score + 15 visual + 15 structural).

## Related Resources

| Resource | Description |
|----------|-------------|
| [poster-sentry (GitHub)](https://github.com/fairdataihub/poster-sentry) | Installable classifier package |
| [poster-sentry (HuggingFace)](https://huggingface.co/fairdataihub/poster-sentry) | Model weights (release 1.0.0) |
| [poster-sentry-training-data (HuggingFace)](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data) | Dataset on HuggingFace Hub |
| [poster-sentry-evaluation-paper-code](https://github.com/fairdataihub/poster-sentry-evaluation-paper-code) | Reproducible analysis for the paper |
| [poster2json](https://github.com/fairdataihub/poster2json) | Poster to structured JSON extraction |
| [posters.science](https://posters.science) | Platform |

## Citation

```bibtex
@dataset{poster_sentry_data_2026,
  title = {PosterSentry Training Data: Scientific Poster Text Corpus},
  author = {O'Neill, Jamey and Portillo, Dorian and Zeinali, Nahid and Soundarajan, Sanjay and Blake, Gerard and Sarin, Parth and Buttrick, Adam and Patel, Bhavesh},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/fairdataihub/poster-sentry-training},
  note = {Part of the posters.science initiative at FAIR Data Innovations Hub}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [FAIR Data Innovations Hub](https://fairdataihub.org/) at California Medical Innovations Institute (CalMI2)
- [posters.science](https://posters.science) platform
- Funded by [The Navigation Fund](https://doi.org/10.71707/rk36-9x79), "Poster Sharing and Discovery Made Easy"
