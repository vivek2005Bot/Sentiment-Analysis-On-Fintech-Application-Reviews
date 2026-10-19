"""Phase 3: data quality report (plan section 7).

Computes quality metrics on the canonical master dataset and writes:
- data/processed/quality_report.json      (machine-readable, full)
- reports/tables/quality_by_bank.csv      (per-bank summary)
- reports/tables/quality_missingness.csv  (missingness per column)
No rows are removed here; issues are measured and reported. Removal is the
loader's job and is always logged (Rule 5).
"""

from __future__ import annotations


import pandas as pd

import src.config as sc
from src.utils import (
    EMOJI_CHAR_CLASS,
    HTML_ENTITY_RE,
    HTML_TAG_RE,
    save_json,
    setup_logging,
)

logger = setup_logging("data_validator")

TEXT_COLS = [
    "review_text",
    "rating",
    "review_date",
    "review_id",
    "reply_text",
    "reply_date",
    "app_version",
]


def _safe_series_regex_count(s: pd.Series, pattern: str) -> pd.Series:
    return s.fillna("").astype("string").str.count(pattern).astype("int32")


def validate_master(master: pd.DataFrame) -> dict:
    """Compute the full quality report (plan section 7 checklist)."""
    cfg = sc.get_config()["quality"]
    n = len(master)
    report: dict = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "total_rows": int(n),
        "rows_per_bank": master["bank"].value_counts().to_dict(),
        "rows_per_bank_type": master["bank_type"].value_counts().to_dict(),
    }

    empty = master["is_empty_text"].astype(bool)
    report["empty_text_rows"] = int(empty.sum())
    report["empty_text_pct"] = round(100 * empty.mean(), 3) if n else 0.0

    dup_ids = master.duplicated(subset=["bank", "review_id"], keep=False)
    report["duplicate_review_ids_rows"] = int(dup_ids.sum())

    norm = (
        master["review_text"]
        .astype("string")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    dup_text = norm.duplicated(keep=False) & norm.notna()
    report["duplicate_normalized_text_rows"] = int(dup_text.sum())
    report["duplicate_text_policy"] = cfg["duplicate_text_action"]

    rating = pd.to_numeric(master["rating"], errors="coerce")
    report["rating_distribution"] = {
        str(int(k)): int(v)
        for k, v in rating.value_counts(dropna=False).sort_index().items()
    }
    report["rating_missing"] = int(rating.isna().sum())

    dates = pd.to_datetime(master["review_date"], errors="coerce")
    report["date_range"] = {
        "min": str(dates.min()),
        "max": str(dates.max()),
        "unparseable_or_missing": int(dates.isna().sum()),
    }
    report["malformed_records"] = {
        "unparseable_dates": int(dates.isna().sum()),
        "non_numeric_ratings": int(rating.isna().sum() - master["rating"].isna().sum()),
        "negative_or_zero_ratings": int((rating <= 0).sum()),
        "ratings_above_five": int((rating > 5).sum()),
    }

    text = master["review_text"].astype("string").fillna("")
    report["text_length_chars"] = {
        "mean": round(float(text.str.len().mean()), 1),
        "median": float(text.str.len().median()),
        "p95": float(text.str.len().quantile(0.95)),
        "max": int(text.str.len().max()),
    }

    artifact_counts = {
        "urls": int(_safe_series_regex_count(text, cfg["url_regex"]).sum()),
        "emails": int(_safe_series_regex_count(text, cfg["email_regex"]).sum()),
        "phone_like": int(_safe_series_regex_count(text, cfg["phone_regex"]).sum()),
        "account_like_digits": int(
            _safe_series_regex_count(text, cfg["long_digit_regex"]).sum()
        ),
        "emojis": int(_safe_series_regex_count(text, EMOJI_CHAR_CLASS).sum()),
        "html_tags": int(text.map(lambda s: bool(HTML_TAG_RE.search(s))).sum()),
        "html_entities": int(text.map(lambda s: bool(HTML_ENTITY_RE.search(s))).sum()),
    }
    report["artifact_counts"] = artifact_counts

    missingness = {
        c: int(master[c].isna().sum()) for c in TEXT_COLS if c in master.columns
    }
    report["missingness"] = missingness
    return report


def run_quality() -> dict:
    """Load master, validate, and persist all quality artifacts."""
    paths = sc.get_paths()
    master = pd.read_parquet(paths["master_parquet"])
    report = validate_master(master)

    by_bank = (
        master.groupby(["bank", "bank_type"], observed=True)
        .agg(
            reviews=("review_id", "size"),
            empty_text=("is_empty_text", "sum"),
            mean_rating=("rating", "mean"),
            min_date=("review_date", "min"),
            max_date=("review_date", "max"),
        )
        .reset_index()
    )
    by_bank["mean_rating"] = by_bank["mean_rating"].round(3)
    by_bank.to_csv(paths["tables_dir"] / "quality_by_bank.csv", index=False)

    miss = pd.DataFrame(
        [
            {
                "column": c,
                "missing": v,
                "missing_pct": round(100 * v / max(len(master), 1), 2),
            }
            for c, v in report["missingness"].items()
        ]
    )
    miss.to_csv(paths["tables_dir"] / "quality_missingness.csv", index=False)

    save_json(report, paths["processed_dir"] / "quality_report.json")
    logger.info(
        "Quality report: total=%d | empty=%d | dup_text=%d",
        report["total_rows"],
        report["empty_text_rows"],
        report["duplicate_normalized_text_rows"],
    )
    return report


if __name__ == "__main__":
    rep = run_quality()
    print(
        {
            k: rep[k]
            for k in [
                "total_rows",
                "empty_text_rows",
                "duplicate_review_ids_rows",
                "duplicate_normalized_text_rows",
            ]
        }
    )
    print(rep["rows_per_bank_type"])
