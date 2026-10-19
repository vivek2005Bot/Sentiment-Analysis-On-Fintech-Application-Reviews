"""Dashboard page-render tests (plan s.48).

Each page script is executed headlessly through Streamlit's AppTest and must
render without raising. This catches import errors, missing-artifact crashes
and pandas/Streamlit API misuse across all 14 pages, not just the entry point.

Pages read the real precomputed predictions (read-only), so they exercise the
actual dashboard code path.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES_DIR = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages"
PAGE_FILES = sorted(PAGES_DIR.glob("*.py"))

EXPECTED_PAGES = {
    "01_overview", "02_sentiment", "03_risk", "04_govt_private",
    "05_language", "06_benchmark", "07_aspects", "08_response",
    "09_investigator", "10_critical_queue", "11_explainability",
    "12_early_warning", "13_issue_explorer", "14_methodology",
}


def test_all_plan_pages_exist():
    """Plan s.48 requires 13 pages; Issue Explorer (s.50) makes 14."""
    found = {p.stem for p in PAGE_FILES}
    assert EXPECTED_PAGES <= found, f"missing pages: {EXPECTED_PAGES - found}"


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.stem)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(page), default_timeout=180)
    at.run()
    assert not at.exception, (
        f"{page.name} raised: "
        f"{[e.value for e in at.exception]}")


def test_no_unmasked_review_text_leaks():
    """Rule 11: dashboard helpers must route review text through mask_pii."""
    src = (Path(__file__).resolve().parents[1] / "streamlit_app" /
           "components" / "common.py").read_text(encoding="utf-8")
    assert "mask_pii" in src, "common.py must use mask_pii for display"
    assert "def pii(" in src
    assert "def review_card(" in src