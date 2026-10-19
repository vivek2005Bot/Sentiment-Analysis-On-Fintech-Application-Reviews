"""Shared pytest fixtures.

Critical safety net (docs/DECISIONS.md D9): engine modules write aggregate
tables into the real `reports/tables/`, `reports/model_results/`, and
`data/*/` directories by default. A unit test built on a 10-row fixture once
overwrote the 12,513-row human annotation sample and the real report tables.

The autouse `isolate_output_dirs` fixture below redirects every configured
output directory into a per-test tmp dir, so running the test suite can never
mutate real project artifacts. Raw data dirs are read-only and untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import src.config as sc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
for _p in (str(PROJECT_ROOT), str(STREAMLIT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Output dirs that engines/tests may write into. Raw data and config are
# deliberately excluded (read-only inputs).
_ISOLATED_KEYS = (
    "annotations_dir",
    "interim_dir",
    "predictions_dir",
    "models_dir",
    "tables_dir",
    "figures_dir",
    "model_results_dir",
    "logs_dir",
    "inventory_dir",
    "processed_dir",
    "samples_dir",
)


@pytest.fixture(autouse=True)
def isolate_output_dirs(tmp_path, monkeypatch):
    """Redirect all engine output paths into tmp so tests are side-effect free."""
    sandbox = tmp_path / "project_io"
    sandbox.mkdir(parents=True, exist_ok=True)
    subdirs = {key: sandbox / key for key in _ISOLATED_KEYS}
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    original_get_paths = sc.get_paths

    def _sandboxed_get_paths():
        paths = dict(original_get_paths())
        paths.update(subdirs)
        return paths

    monkeypatch.setattr(sc, "get_paths", _sandboxed_get_paths)
    yield subdirs
