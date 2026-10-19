"""Master pipeline orchestrator (plan Phases 4-18, output files s.54).

Runs the full analysis stack over master_reviews.parquet:
preprocessing -> language -> sentiment -> failure gate -> risk -> emotion ->
aspects -> response features, then writes the master predictions parquet and
the aggregate report tables. Deterministic engines only; transformer
inference (when models exist) is merged by model_registry before final write.
"""

from __future__ import annotations

import pandas as pd

import src.config as sc
from src import (
    aspect_engine,
    emotion_engine,
    failure_detector,
    language_detector,
    preprocessing,
    risk_engine,
    response_engine,
    sentiment_engine,
    statistical_analysis,
)
from src.pagerank_engine import compute_pagerank
from src.utils import setup_logging

logger = setup_logging("pipeline")


def run_full_pipeline(sample_n: int | None = None) -> pd.DataFrame:
    paths = sc.get_paths()
    master = pd.read_parquet(paths["master_parquet"])
    if sample_n:
        master = master.sample(
            n=min(sample_n, len(master)), random_state=sc.get_seed()
        ).reset_index(drop=True)
        logger.info("PIPELINE SAMPLE RUN: %d rows", len(master))
    logger.info("Pipeline start: %d rows", len(master))

    # Phase 4: model text + emoji metadata (raw text preserved)
    master = preprocessing.add_text_fields(master)

    # Phase 5: language/script/Hinglish
    master = language_detector.detect_language_frame(master)

    # Sentiment baseline v0 (rules; transformers replace via model_registry)
    master = sentiment_engine.score_sentiment_frame(master)

    # Tier-1 failure gate
    master = failure_detector.detect_failure_frame(master)

    # Tier-2 risk + severity
    master = risk_engine.classify_risk_frame(master)

    # Emotion
    master = emotion_engine.detect_emotion_frame(master)

    # Aspects + aspect sentiment
    master = aspect_engine.detect_aspects_frame(master)

    # Response features
    master = response_engine.add_response_features(master)

    # Rating/sentiment mismatch flag (s.29)
    master = sentiment_engine.rating_sentiment_mismatch(master)

    out = paths["predictions_dir"] / "master_predictions.parquet"
    master.to_parquet(out, index=False)
    logger.info("Master predictions: %d rows -> %s", len(master), out)

    # Plan s.54: per-task prediction files + aspect edge list
    from src import export

    export.write_task_predictions(master)
    export.write_aspect_edges(master)

    if not sample_n:  # aggregates only on the full corpus
        build_report_tables(master)
    return master


def build_report_tables(master: pd.DataFrame) -> None:
    """Write the aggregate report tables (plan s.55 Tables 1-7 subset)."""
    paths = sc.get_paths()

    # Table 1: dataset
    t1 = (
        master.groupby(["bank", "bank_type"], observed=True)
        .agg(
            reviews=("review_id", "size"),
            avg_rating=("rating", "mean"),
            min_date=("review_date", "min"),
            max_date=("review_date", "max"),
        )
        .reset_index()
    )
    t1["avg_rating"] = t1["avg_rating"].round(3)
    t1.to_csv(paths["tables_dir"] / "bank_summary.csv", index=False)

    # Table 2: language
    t2 = (
        master.groupby("language", dropna=False, observed=True)
        .agg(
            reviews=("review_id", "size"),
            negative_rate=("sentiment", lambda s: float((s == "NEGATIVE").mean())),
            failure_rate=("failure_flag", lambda s: float((s == "YES").mean())),
        )
        .reset_index()
    )
    t2["percentage"] = (100 * t2["reviews"] / len(master)).round(3)
    t2["negative_rate"] = t2["negative_rate"].round(4)
    t2["failure_rate"] = t2["failure_rate"].round(4)
    t2.to_csv(paths["tables_dir"] / "language_summary.csv", index=False)

    # Table 4: risk by bank
    t4 = (
        master.groupby(["bank", "bank_type"], observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "reviews": len(g),
                    "failure_rate": (g["failure_flag"] == "YES").mean(),
                    "technical_risk_rate": (
                        (g["primary_risk"] == "TECHNICAL_PERFORMANCE_RISK").mean()
                    ),
                    "financial_risk_rate": (
                        (g["primary_risk"] == "FINANCIAL_SECURITY_RISK").mean()
                    ),
                    "friction_rate": (g["operational_friction"] == "YES").mean(),
                    "critical_failure_rate": g["critical_failure"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    for c in t4.columns[2:]:
        t4[c] = t4[c].astype(float).round(4)
    t4.to_csv(paths["tables_dir"] / "risk_by_bank.csv", index=False)

    # Table 5: Govt vs Private + stats
    gvp = statistical_analysis.govt_vs_private_table(master)
    gvp.to_csv(paths["tables_dir"] / "govt_vs_private.csv", index=False)

    # Table 7: response
    resp = response_engine.response_summary_table(master)
    resp.to_csv(paths["tables_dir"] / "response_analysis.csv", index=False)

    # PageRank tables
    compute_pagerank(master)

    # Temporal trends + early warning + improvement matrix
    from src import export, temporal_engine

    temporal_engine.monthly_trends(master)
    temporal_engine.early_warning(master)
    temporal_engine.bank_improvement_matrix(master)

    # Consolidated PageRank-by-bank table + model benchmark table (s.54/s.55)
    export.write_pagerank_by_bank()
    export.write_model_benchmark()

    # Required visualizations (plan s.56)
    import src.figures as figures

    figures.build_all_figures(master)

    logger.info("Report tables written to %s", paths["tables_dir"])


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = run_full_pipeline(sample_n=n)
    print("\n=== PIPELINE COMPLETE ===")
    print(f"rows: {len(df)}")
    print("sentiment:", df["sentiment"].value_counts().to_dict())
    print("failure:", df["failure_flag"].value_counts(dropna=False).to_dict())
    print("primary_risk:", df["primary_risk"].value_counts(dropna=False).to_dict())
    print("language:", df["language"].value_counts(dropna=False).to_dict())
    print("severity:", df["severity"].value_counts().sort_index().to_dict())
