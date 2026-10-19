#!/bin/bash
# End-to-end reproducible pipeline (offline stages; training optional).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== 1/5 Data inventory + canonical master ==="
$PY -m src.data_loader

echo "=== 2/5 Full analysis pipeline + report tables ==="
$PY -m src.pipeline

echo "=== 3/5 Annotation sample + silver labels ==="
$PY -m src.annotation

echo "=== 4/5 Unit tests ==="
$PY -m pytest tests/ -q

echo "=== 5/5 Optional: transformer smoke benchmark (downloads weights) ==="
$PY -m src.train --task sentiment --model distilbert-base-uncased --smoke || \
  echo "(smoke training skipped/failed - see logs/train_smoke.log)"

echo "ALL DONE. Dashboard: .venv/bin/streamlit run streamlit_app/app.py"
