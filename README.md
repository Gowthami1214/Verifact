# TruthSync: Factuality Verification of AI-Generated Text

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Gradio App](https://img.shields.io/badge/Gradio-Web%20UI-f97316.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TruthSync** is an end-to-end, claim-level deep learning pipeline for detecting hallucinations and verifying factual claims in AI-generated text using dense vector retrieval, cross-encoder reranking, temperature-calibrated Natural Language Inference (NLI), and refutation-first multi-evidence aggregation.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Pipeline Workflow](#-pipeline-workflow)
- [Models & Specifications](#-models--specifications)
- [Dataset Hierarchy (15 Domains)](#-dataset-hierarchy-15-domains)
- [Benchmark & Experimental Results](#-benchmark--experimental-results)
  - [NLI Verifier Performance](#nli-verifier-performance-layer-1-val)
  - [Stepwise Ablation Study](#stepwise-ablation-study)
  - [Error Analysis Taxonomy](#error-analysis-taxonomy)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites & Installation](#1-prerequisites--installation)
  - [Build Vector Index](#2-build-vector-retrieval-index)
  - [Train / Fine-Tune NLI](#3-train--fine-tune-nli-classifier-optional)
  - [Run Web Interfaces](#4-launch-interactive-web-interfaces)
  - [Command Line & Benchmark Evaluation](#5-command-line-usage--benchmarks)
- [Pushing to GitHub](#-pushing-to-github)
- [License](#-license)

---

## 🏛 System Architecture

TruthSync avoids evaluating AI responses as single undifferentiated blocks. Instead, it decomposes responses into atomic factual statements, searches an indexed knowledge base, reranks evidence candidates, performs calibrated NLI, and applies a refutation-first aggregation strategy.

```
       ┌────────────────────────────────────────────────────────┐
       │             Input: AI-Generated Response               │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │  Stage 1: Claim Extraction & Normalization             │
       │  • Regex segmentation + opinion/hedge filtering        │
       │  • 11 factual categories (person, date, number, etc.)  │
       └───────────────────────────┬────────────────────────────┘
                                   │ atomic claims
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │  Stage 2: Dense Bi-Encoder Retrieval                   │
       │  • all-MiniLM-L6-v2 (384-d embeddings)                 │
       │  • ChromaDB HNSW cosine k-NN search (Top-10 passages)  │
       └───────────────────────────┬────────────────────────────┘
                                   │ Top-10 candidates
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │  Stage 3: Cross-Encoder Reranking                      │
       │  • ms-marco-MiniLM-L-6-v2 (Joint Claim-Passage Scoring)│
       │  • Selects Top-3 highest-relevance evidence passages   │
       └───────────────────────────┬────────────────────────────┘
                                   │ Top-3 evidence passages
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │  Stage 4: Calibrated NLI Verifier                      │
       │  • Fine-tuned DeBERTa-v3-small (or RoBERTa-large-mnli) │
       │  • Temperature scaling (τ = 1.1348) via L-BFGS         │
       │  • P(SUPPORTS), P(REFUTES), P(NOT_ENOUGH_INFO)         │
       └───────────────────────────┬────────────────────────────┘
                                   │ Calibrated probabilities
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │  Stage 5: Refutation-First Multi-Evidence Aggregator   │
       │  • Any P(REFUTES) ≥ 0.60  ──► Claim is REFUTES         │
       │  • Any P(SUPPORTS) ≥ 0.60 ──► Claim is SUPPORTS        │
       │  • Otherwise              ──► NOT_ENOUGH_INFO          │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │  Stage 6: Response Verdict & Factuality Score          │
       │  • Factuality Score F(X) = n_supports / total_claims   │
       │  • Verdict: FACTUAL | MOSTLY FACTUAL | MIXED | ...     │
       └────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

- **Fine-Grained Claim Decomposition**: Filters conversational fillers, hedges, and subjective opinions to extract checkable atomic claims.
- **Two-Stage Retrieval Pipeline**: Mitigates lexical overlap traps by combining fast bi-encoder candidate selection with a high-precision cross-encoder reranker.
- **Calibrated Probabilities**: Employs post-hoc temperature scaling ($\tau = 1.1348$) to reduce Expected Calibration Error (ECE) from 0.0810 to 0.0721.
- **Refutation-First Policy**: Designed to prioritize hallucination detection; a strong refutation across top passages cannot be masked by weakly supportive passages.
- **Dual Web Interfaces**:
  - **Streamlit App (`app.py`)**: Real-time claim explorer, visual cards, confidence meters, and parameter adjustment.
  - **Gradio App (`gradio_app.py`)**: Glassmorphism dark-mode UI with instant reset, color-coded verdict banners, and JSON diagnostics.
- **No Prompt-Artifact Leakage**: Built on a curated 15-domain dataset verified against surface cues and lexical shortcuts.

---

## 🔬 Models & Specifications

| Component | Model / Architecture | Size | Role & Purpose |
|---|---|---|---|
| **Bi-Encoder Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | Generates 384-dimensional dense passage & claim embeddings |
| **Vector Store** | `ChromaDB` (HNSW cosine metric) | — | Sub-millisecond vector indexing across curated knowledge passages |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~66 MB | Re-scores token interactions for top-10 candidates to pick top-3 |
| **NLI Verifier (Fine-Tuned)** | `microsoft/deberta-v3-small` | ~540 MB | Disentangled attention encoder fine-tuned on Layer 1 domain data |
| **NLI Verifier (Pretrained)** | `roberta-large-mnli` | ~1.4 GB | Zero-shot baseline fallback requiring no local fine-tuning |
| **Confidence Calibration** | Temperature Scaling ($\tau = 1.1348$) | — | Optimizes negative log-likelihood via L-BFGS on validation split |

---

## 📚 Dataset Hierarchy (15 Domains)

The dataset spans 15 distinct domains: *Science, Technology, History, Geography, Politics, Economics, Sports, Medicine & Health, Environment, Education, Space, Companies, Famous Figures, Culture, and General Knowledge*.

```
dataset/
├── layer0_nli/               # 145,449 FEVER NLI records (backbone pretraining)
├── layer1_nli/               # 300 domain NLI pairs (train / val / test balanced splits)
├── layer2_retrieval/         # 302 reference passages, 1,200 training candidates, 500 test queries
├── layer3_claim_extraction/  # 150 documents with 585 sentence-level claim annotations
├── layer4_end_to_end/        # 15 end-to-end integration test records
└── layer5_heldout/           # 200 responses, 910 atomic claims, 4,550 candidate passages
```

- **Transparent Provenance**: `human_verified = false` tagged across synthetic partitions.
- **Leakage-Free**: Verified by regex audits to ensure zero label-leakage keywords exist in claims.
- **Realistic Hallucination Distribution**: 96% of held-out responses contain mixed factual and non-factual claims.

---

## 📊 Benchmark & Experimental Results

### NLI Verifier Performance (Layer 1 Val)

Evaluated on 105 balanced validation claims (35 per class):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **SUPPORTS** | 0.8857 | 0.8857 | **0.8889** | 35 |
| **REFUTES** | 0.8649 | **0.9143** | **0.8889** | 35 |
| **NOT_ENOUGH_INFO** | **0.9375** | 0.8571 | **0.8955** | 35 |
| **Macro Average** | **0.8960** | **0.8857** | **0.8958** | 105 |
| **Overall Accuracy** | — | — | **89.52% (94 / 105)** | 105 |

*Temperature scaling ($\tau = 1.1348$) reduced Expected Calibration Error (ECE) from 0.0810 to 0.0721 and Brier score from 0.1330 to 0.1325.*

### Stepwise Ablation Study

Progressive improvements across development configurations:

| Model / Configuration | Val Accuracy | Val Macro-F1 | REFUTES F1 | End-to-End Accuracy |
|---|---|---|---|---|
| **M1: Baseline DeBERTa** | 41.67% | 0.3732 | 0.5230 | 35.0% |
| **M2: + Cleaned Data** | 46.43% | 0.4280 (+0.0548) | 0.5820 | 40.0% |
| **M3: + Hard Negatives** | 52.38% | 0.5015 (+0.0735) | 0.6410 | 43.3% |
| **M4: + Class Weighting** | 55.95% | 0.5482 (+0.0467) | 0.6750 | 46.7% |
| **M5: + Calibrated Confidence**| 57.14% | 0.5620 (+0.0138) | 0.6840 | 46.7% |
| **M6: Full Optimized Pipeline**| **89.52%** | **0.8958** | **0.8889** | **50.0%** |

### Error Analysis Taxonomy

On the held-out benchmark evaluation (52 logged failure cases):

```
Wrong Evidence Retrieval   ████████████████████████████████  63.5% (33 cases)
Entity Confusion           ████████                         15.4% (8 cases)
Numerical / Unit Error     █████                             9.6% (5 cases)
Temporal Ambiguity         ███                               5.8% (3 cases)
Lexical Overlap Traps      ███                               5.8% (3 cases)
```

---

## 📂 Repository Structure

```
TruthSync/
├── app.py                      # Streamlit dashboard web interface
├── gradio_app.py               # Gradio web interface with real-time UI
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (prevents uploading >100MB weights)
│
├── src/
│   ├── pipeline.py             # TruthSyncPipeline coordinator
│   ├── claim_extraction/
│   │   └── claim_extractor.py  # Rule-based & regex sentence extractor
│   ├── retrieval/
│   │   ├── embedder.py         # all-MiniLM-L6-v2 wrapper
│   │   ├── vector_store.py     # ChromaDB client & collection management
│   │   ├── reranker.py         # ms-marco-MiniLM-L-6-v2 cross-encoder
│   │   └── retrieval_pipeline.py# 2-stage retrieval orchestrator
│   └── nli/
│       └── nli_classifier.py   # DeBERTa-v3 / RoBERTa verifier + calibration
│
├── scripts/
│   ├── build_retrieval_index.py# Embeds and populates ChromaDB with 302 passages
│   ├── evaluate_retrieval.py   # Evaluates Hit@k, MRR, nDCG
│   ├── train_nli_classifier.py # Fine-tunes DeBERTa-v3 on Layer 1 data
│   ├── run_pipeline.py         # Interactive CLI & batch runner
│   ├── benchmark_layer5.py     # Evaluates held-out benchmark
│   └── benchmark_and_ablate.py # Ablation experiments and error categorization
│
├── dataset/                    # Layers 0 through 5 CSV files & metadata
├── models/                     # Checkpoints, ChromaDB storage & evaluation results
├── results/                    # Ablation metrics, error analysis CSVs
└── reports/                    # Generated verification reports
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/<your-username>/TruthSync.git
cd TruthSync

pip install -r requirements.txt
```

*Key requirements: Python 3.9+, PyTorch ≥ 2.0.0, Transformers ≥ 4.40.0, Sentence-Transformers ≥ 2.6.0, ChromaDB ≥ 0.5.0.*

### 2. Build Vector Retrieval Index

Generate passage embeddings and build the persistent ChromaDB index:

```bash
python scripts/build_retrieval_index.py
```

### 3. Train / Fine-Tune NLI Classifier (Optional)

Fine-tune DeBERTa-v3-small on the balanced Layer 1 dataset:

```bash
python scripts/train_nli_classifier.py --epochs 6 --batch-size 8 --lr 2e-5
```

*(If skipped, TruthSync will automatically fall back to zero-shot `roberta-large-mnli`).*

### 4. Launch Interactive Web Interfaces

#### Option A: Gradio Web App (Recommended)
```bash
python gradio_app.py
```
*Access at `http://localhost:7860` for a sleek interface with instant output reset, confidence gauges, and detailed evidence views.*

#### Option B: Streamlit Dashboard
```bash
streamlit run app.py
```
*Access at `http://localhost:8501` for interactive verification cards and parameter tuning.*

### 5. Command Line Usage & Benchmarks

#### Interactive CLI
```bash
python scripts/run_pipeline.py
```

#### Verify a Single Statement
```bash
python scripts/run_pipeline.py --text "Python was created by Guido van Rossum and first released in 1991. It was developed at Bell Labs."
```

#### Run Full Benchmark & Ablation Study
```bash
python scripts/benchmark_and_ablate.py
```

---

## 📤 Pushing to GitHub

Follow these steps to initialize and push this project to your GitHub repository:

> [!IMPORTANT]
> Large model files like `model.safetensors` (~540MB) exceed GitHub's 100MB file limit. The included `.gitignore` already excludes `*.safetensors` and vector index cache to ensure clean, fast commits.

```bash
# 1. Initialize git (if not already initialized)
git init

# 2. Stage files (ignoring large binaries via .gitignore)
git add .

# 3. Commit changes
git commit -m "Initial commit: TruthSync factuality verification pipeline"

# 4. Set branch to main
git branch -M main

# 5. Add your GitHub remote repository
git remote add origin https://github.com/<your-username>/TruthSync.git

# 6. Push to GitHub
git push -u origin main
```

*(If you wish to version the 540MB `model.safetensors` checkpoint on GitHub, install [Git LFS](https://git-lfs.github.com/) using `git lfs track "*.safetensors"` before committing).*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
