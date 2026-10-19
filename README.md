# Indian Banking Review Intelligence 🏦📊

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14+-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-5.17+-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/pytest-53%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An enterprise-grade Natural Language Processing (NLP) intelligence platform and interactive executive dashboard for analyzing Indian banking application reviews from the Google Play Store (~449,000+ reviews).

---

## ⚡ Quickstart: Running the Dashboard

### 1. Activate the Virtual Environment
Before running commands, activate the Python environment:
```bash
source .venv/bin/activate
```
*(If the virtual environment is not yet installed, install dependencies with: `pip install -r requirements.txt`)*

### 2. Launch Streamlit
Run the command pointing to `streamlit_app/app.py` from the project root:
```bash
streamlit run streamlit_app/app.py
```

> [!TIP]
> **Encountered `zsh: command not found: streamlit`?**
> This happens when your terminal session hasn't activated `.venv`. You can either run `source .venv/bin/activate` first or directly invoke the binary:
> ```bash
> .venv/bin/streamlit run streamlit_app/app.py
> ```

---

## 🚀 System Architecture & Capabilities

```
User Review ────────► [ Phase 1: Preprocessing & PII Masking ]
                             │
                             ▼
                      [ Phase 2: Language & Script Detection ]
                      (English, Hindi, Hinglish, Code-mixed)
                             │
                             ▼
                      [ Phase 3: Sentiment Engine ]
                      (Prior-weighted lexicon + DistilBERT)
                             │
                             ▼
                      [ Phase 4: Tier-1 Failure Gate ]
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
    [ Failure Detected ]            [ Operational Friction ]
             │                       (UI lag, feature requests)
             ▼
   [ Tier-2 Risk Classifier ]
   ├── FINANCIAL_SECURITY_RISK (Fraud, unauthorized debits, OTP leaks)
   └── TECHNICAL_PERFORMANCE_RISK (App crash, server timeout, blank screen)
```

1. **Language & Hinglish Intelligence**: Detects Romanized Hindi (Hinglish), Devanagari script, and code-mixed patterns.
2. **Two-Tier Risk Architecture**: 
   - **Tier 1 (Failure Gate)** separates benign complaints from critical system failures.
   - **Tier 2 (Risk Classification)** categorizes severe incidents into **Financial Security Risk** vs. **Technical Performance Risk**.
3. **Aspect-Based Intelligence**: Maps granular issues to banking domains (UPI, Authentication, Biometrics, UI, Customer Support).
4. **Graph PageRank Engine**: Constructs semantic similarity graphs to surface the most representative customer voices per bank.
5. **Early Warning Signals & Anomaly Detection**: Tracks temporal spike anomalies and version regressions.
6. **Executive Streamlit Decision Dashboard**: 14 analytical pages with interactive KPIs, Plotly charts, and audit queues.

---

## 📁 Project Structure

```
├── configs/                  # Project and model hyperparameters
│   ├── config.yaml           # Canonical paths and global parameters
│   └── model_config.yaml     # Engine weights, thresholds, and lexicons
├── data/
│   ├── inventory/            # File manifest and data dictionary
│   ├── processed/            # Master clean dataset (master_reviews.parquet)
│   ├── predictions/          # NLP engine prediction parquets
│   ├── annotations/          # Stratified human annotation queue & silver labels
│   └── samples/              # 2K demo sandbox datasets
├── docs/
│   └── DECISIONS.md          # Architectural Decision Records (ADRs)
├── Govt_Banks/               # Raw review data for Public Sector banks (immutable)
├── Pvt_Banks/                # Raw review data for Private Sector banks (immutable)
├── logs/                     # Application execution logs (logs/.gitkeep)
├── models/                   # Fine-tuned Transformer checkpoints & label maps
├── notebooks/                # Exploratory Data Analysis (EDA) Jupyter notebooks
│   └── FDS - PROJECT.ipynb   # Initial scraping & exploratory analysis
├── reports/
│   ├── figures/              # Precomputed high-resolution analytical figures
│   ├── tables/               # 29 aggregate CSV decision tables
│   └── model_results/        # Transformer benchmark logs and evaluations
├── scripts/
│   └── run_all.sh            # End-to-end reproducible pipeline script
├── src/                      # Core NLP and Data Science engines
│   ├── data_loader.py        # Schema normalization and inventory creation
│   ├── preprocessing.py      # Cleaners, contractions, and PII masking
│   ├── language_detector.py  # Language and Hinglish identification
│   ├── sentiment_engine.py   # Sentiment classification
│   ├── failure_detector.py   # Tier-1 failure and friction engine
│   ├── risk_engine.py        # Tier-2 risk classifier
│   ├── emotion_engine.py     # Emotion extraction
│   ├── aspect_engine.py      # Aspect classification
│   ├── temporal_engine.py    # Time-series and anomaly tracking
│   ├── pagerank_engine.py    # Review graph centrality
│   ├── statistical_analysis.py # Statistical comparison and aggregation
│   ├── pipeline.py           # Master execution pipeline
│   ├── train.py              # Transformer training on MPS/CUDA
│   └── evaluate.py           # Model evaluation and metrics
├── streamlit_app/            # Interactive Multi-Page Web Dashboard
│   ├── app.py                # Main entrypoint and navigation
│   ├── components/           # Reusable UI widgets and Plotly charts
│   └── pages/                # 14 distinct analytical intelligence views
├── tests/                    # Pytest test suite (53 unit & dashboard tests)
├── PROJECT_GUIDE.md          # In-depth operational project manual
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python package dependencies
└── README.md                 # Project overview and quickstart
```

---

## 🛠️ Offline Pipeline Execution

To run the complete data processing, inference, and reporting pipeline from scratch:

```bash
# Execute end-to-end pipeline with one command
bash scripts/run_all.sh
```

Or execute modular stages independently:

```bash
# 1. Standardize raw files & build canonical master parquet
python -m src.data_loader

# 2. Run NLP analysis stack & generate summary report tables
python -m src.pipeline

# 3. Generate stratified annotation samples and silver labels
python -m src.annotation

# 4. Run full test suite
pytest
```

---

## 📊 Dataset Corpus Overview

| Category | Banks | Reviews Count |
|:---|:---|:---:|
| **Public Sector (Government)** | State Bank of India (SBI), Union Bank, UCO Bank, Indian Bank, Bank of India, Central Bank of India, Indian Overseas Bank, Bank of Maharashtra | ~418,800 |
| **Private Sector** | IndusInd Bank (~60.4K), Axis Bank (~9.2K), ICICI Bank (~1.0K) | ~70,600 |
| **Reported Empty Sources** | HDFC Bank, Kotak Mahindra, IDFC First, Federal Bank, Bandhan Bank, City Union Bank | 0 *(documented, unflattened)* |
| **Total Canonical Master** | **11 Banks** | **449,379** |

---

## 🧪 Testing & Verification

Run the comprehensive unit test suite:
```bash
pytest
```
*53 passed tests verifying data loading, NLP engines, risk rules, PageRank, and dashboard components.*

---

## 🔒 Ethical & Honest-Model Principles

- **No Data Fabrication**: Empty bank files are recorded transparently as unusable rather than synthetically filled.
- **Silver Label Distinction**: Lexicon/heuristic outputs are labeled as `silver_rules_v1` and separated from human gold labels.
- **Privacy First (PII Masking)**: All phone numbers, emails, card numbers, and account identifiers are strictly masked prior to rendering in dashboard views.
- **Observational Findings**: Group differences (Public vs. Private) are reported descriptively without unsubstantiated causal claims.

---

## 📜 License
This project is licensed under the MIT License.
