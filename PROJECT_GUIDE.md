# Indian Banking Review Intelligence - Project Guide

## What is this Project About?
This project is an industry-level Natural Language Processing (NLP) and FinTech review-intelligence system. It processes and analyzes a massive corpus of Google Play Store reviews for various **Government** and **Private** banks in India (approx 449k reviews).

The system uses rules, lexicons, and Transformer-based models to extract structured insights from unstructured user reviews. Key capabilities include:
- **Language Detection:** Identifies languages and code-mixing (e.g., Hinglish).
- **Sentiment Analysis:** Classifies reviews as positive, neutral, or negative.
- **Failure & Friction Detection:** Identifies critical app failures (e.g., crashes) and operational friction (e.g., slow UI).
- **Risk Classification:** Detects financial security risks (e.g., fraud, lost money) vs. technical performance risks.
- **Aspect & Emotion Extraction:** Identifies what feature the user is talking about (UI, UPI, login) and their emotion.
- **Dashboard:** An interactive Streamlit dashboard to visualize all the processed intelligence.

---

## How to Run the Project

### Prerequisites
Make sure your Python virtual environment is activated:
```bash
source .venv/bin/activate
```

### 1. Running the Full Pipeline (Offline)
You can run the entire data processing, analysis, annotation, and testing pipeline with a single shell script:
```bash
bash scripts/run_all.sh
```

Alternatively, you can run the pipeline step-by-step:
```bash
# 1. Load data, build inventory, and create master dataset
.venv/bin/python -m src.data_loader          

# 2. Run the NLP engines (sentiment, risk, etc.) and generate report tables
.venv/bin/python -m src.pipeline             

# 3. Generate human annotation samples and silver labels
.venv/bin/python -m src.annotation           

# 4. Run the Unit Tests
.venv/bin/python -m pytest tests/ -q         
```

### 2. Running the Streamlit Dashboard
The web dashboard provides interactive visualizations of the processed review data.
*Note: Ensure you run the `streamlit` command from the root folder so it resolves paths correctly.*

```bash
.venv/bin/streamlit run streamlit_app/app.py
```
*(If you navigate into the `streamlit_app` folder, use `../.venv/bin/streamlit run app.py` instead).*

---

## Project Structure: Files and Folders

### Root Directories
- **`Govt_Banks/` & `Pvt_Banks/`**: Immutable raw data folders containing the original scraped reviews (CSV/Excel).
- **`data/`**: Stores all intermediate and final data artifacts.
  - `data/processed/`: Contains the canonical clean data (`master_reviews.parquet`).
  - `data/predictions/`: Output from NLP engines.
  - `data/annotations/`: Samples prepared for human annotation.
  - `data/inventory/`: Manifest of loaded raw files.
- **`configs/`**: Project configurations.
  - `config.yaml`: Core project paths and settings.
  - `model_config.yaml`: Settings, weights, and rules for the NLP models.
- **`src/`**: The core Python package containing all logic and NLP engines.
- **`scripts/`**: Shell scripts for automation (e.g., `run_all.sh`).
- **`streamlit_app/`**: The interactive web dashboard codebase.
- **`tests/`**: Pytest unit test suite for ensuring logic correctness.
- **`reports/`**: Generated CSV tables and analytical outputs for business use.
- **`logs/`**: Execution logs for debugging.
- **`models/`**: Stored machine learning models and fine-tuned weights.
- **`docs/`**: Project documentation (e.g., `DECISIONS.md` explaining architectural decisions).
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA).

### Core Engine Files (`src/`)
- **`data_loader.py`**: Discovers raw files, standardizes schemas, and builds the canonical master dataset.
- **`pipeline.py`**: The orchestrator that chains all engines together sequentially.
- **`preprocessing.py`**: Cleans text, handles emojis, and standardizes formats.
- **`language_detector.py`**: Identifies the language of the review (English, Hindi, Hinglish, etc.).
- **`sentiment_engine.py`**: Determines if the review is positive, negative, or neutral.
- **`failure_detector.py`**: Flags operational friction and critical technical failures.
- **`risk_engine.py`**: Classifies severe issues into financial vs technical risk.
- **`emotion_engine.py`**: Extracts primary user emotions (e.g., angry, happy, frustrated).
- **`aspect_engine.py`**: Identifies the aspect/feature mentioned (e.g., UPI, UI, Login).
- **`temporal_engine.py`**: Analyzes trends over time and generates early warning signals.
- **`pagerank_engine.py`**: Graph-based algorithm to find the most representative reviews.
- **`statistical_analysis.py`**: Generates report tables and summary statistics.
- **`annotation.py`**: Prepares a stratified sample of the data for human annotators.
- **`train.py` & `evaluate.py`**: Scripts for fine-tuning and evaluating Transformer models.
- **`config.py` & `utils.py`**: Shared utilities and configuration loaders.

### Dashboard Files (`streamlit_app/`)
- **`app.py`**: The main entry point for the Streamlit dashboard.
- **`pages/`**: Sub-pages for different analytical views (e.g., Risk, Sentiment, Failures).
- **`components/`**: Reusable UI components for the dashboard.
- **`utils/`**: Helper functions specific to Streamlit.

### Root Files
- **`README.md`**: High-level overview and quickstart guide.
- **`INDIAN_BANKING_NLP_IMPLEMENTATION_PLAN.md`**: The master architectural contract and plan.
- **`requirements.txt`**: Python dependencies required to run the project.
- **`pyrightconfig.json`**: Type-checking configuration for Pyright.
- **`run_all.sh`**: The end-to-end pipeline execution script.
