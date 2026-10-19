# Indian Banking Review Intelligence

Transformer-based hierarchical risk & sentiment intelligence for Indian
banking app reviews (Google Play Store data). Implements
`INDIAN_BANKING_NLP_IMPLEMENTATION_PLAN.md` as the master contract.

## Corpus (measured, not assumed)

| Group | Banks | Reviews |
|---|---|---|
| Government | SBI, Union, UCO, Indian Bank, Bank of India, Central Bank of India, Indian Overseas, Bank of Maharashtra | ~418K |
| Private | IndusInd (60.4K), Axis (9.2K), ICICI (1.0K) | ~70.6K |
| Private (EMPTY files, reported not fabricated) | HDFC, Kotak, IDFC First, Federal, Bandhan, City Union | 0 |
| **Total (canonical master)** | **11 banks** | **449,379** |

Every inventory row count reconciles exactly against pandas row counts
(verified). `wc -l` over-reports because review text contains embedded
newlines - see `docs/DECISIONS.md` D10.

Raw folders `Govt_Banks/` and `Pvt_Banks/` are **immutable**.

## Quickstart

```bash
.venv/bin/python -m src.data_loader          # inventory + master parquet
.venv/bin/python -m src.pipeline             # full engine stack + report tables
.venv/bin/python -m src.annotation           # stratified annotation + silver labels
.venv/bin/python -m src.annotation --demo     # 2K-row sandbox, never touches data/annotations/
.venv/bin/python -m pytest tests/ -q         # unit tests
.venv/bin/streamlit run streamlit_app/app.py # dashboard
```

Or the whole offline pipeline: `bash scripts/run_all.sh`

## Architecture (plan sections 12-14)

```
Review -> Tier 1: Failure Gate
            NO  -> Operational friction / normal
            YES -> Tier 2: TECHNICAL_PERFORMANCE_RISK vs FINANCIAL_SECURITY_RISK
                   (financial wins on overlap; secondary signal + risk_overlap kept)
```

## Key outputs

- `data/processed/master_reviews.parquet` - canonical dataset + provenance
- `data/predictions/master_predictions.parquet` - all engine outputs
- `data/annotations/annotation_sample.csv` - human annotation queue (labels blank)
- `reports/tables/*.csv` - bank summary, govt_vs_private (+ stats), language,
  risk, response, PageRank, temporal trends, early warning, improvement matrix
- `reports/model_results/` - transformer benchmark experiments
  (`benchmark_log.csv` is deduplicated: one row per model+task+split+seed+size)
- `reports/figures/` - 22 required visualizations (plan s.56), all generated
  offline by `src.figures` (dashboard reads precomputed PNGs only)

## Transformer benchmark (measured, smoke runs on MPS)

| Task | Model | Accuracy | Macro-F1 | Test n |
|---|---|---|---|---|
| sentiment (3-class) | distilbert-base-uncased | 0.755 | 0.4936 | 200 |
| flat 4-class baseline (s.24) | distilbert-base-uncased | 0.920 | 0.3194 | 200 |

Flat vs hierarchical on the same 2,503 held-out rows
(`reports/tables/flat_vs_hierarchical.csv`): flat_4class macro-F1 = 0.2391,
financial recall = 0.0 - the flat head misses the rare FINANCIAL_RISK class
(36/12,513 silver rows). The hierarchical side has no trained transformer
heads yet, so no hierarchical transformer row is reported (Rule 9: no
fabricated rows). Silver-label benchmarks are provisional until human labels
exist.

## Honest-model rules enforced

- Rule-based labels are **bootstrap silver labels only** (`label_source`
  column) - never presented as human annotations.
- Transformer claims require held-out evaluation (`src/train.py` writes
  experiment metrics to `reports/model_results/`).
- All comparisons are observational; no causal language.
- Every dropped row is logged in `data/processed/removal_log.csv`.
- Dashboard text is PII-masked before display.
