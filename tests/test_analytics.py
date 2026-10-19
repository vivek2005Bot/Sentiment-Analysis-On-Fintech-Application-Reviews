"""Unit tests for analytics engines, statistical tests, and data validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.config as sc
from src import (
    annotation,
    aspect_engine,
    data_validator,
    emotion_engine,
    failure_detector,
    language_detector,
    model_registry,
    pagerank_engine,
    preprocessing,
    response_engine,
    risk_engine,
    sentiment_engine,
    statistical_analysis,
    temporal_engine,
)


@pytest.fixture
def sample_master_df():
    df = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(10)],
        "review_text": [
            "App is crashing every time and money was deducted but not credited!",
            "Very good and fast UPI experience, loved the UI.",
            "Worst customer service, nobody is responding to my complaint.",
            "यह बहुत अच्छा ऐप है धन्यवाद",
            "app romba slow ah irukku please fix this",
            "Unable to login, OTP not received",
            "Superb banking app, smooth transfer and quick response",
            "Money debited twice for single transaction",
            "Nice interface, easy to use",
            "Pathetic server down issue every weekend",
        ],
        "rating": [1, 5, 1, 5, 2, 1, 5, 1, 4, 1],
        "review_date": [
            "2025-01-10 10:00:00", "2025-01-15 11:00:00",
            "2025-02-01 12:00:00", "2025-02-15 13:00:00",
            "2025-03-01 14:00:00", "2025-03-10 15:00:00",
            "2025-03-15 16:00:00", "2025-04-01 10:00:00",
            "2025-04-10 11:00:00", "2025-04-15 12:00:00",
        ],
        "reply_text": [
            "We are sorry", None, "Contact support", None,
            None, "Please retry", None, "Issue noted", None, "Under maintenance"
        ],
        "reply_date": [
            "2025-01-10 18:00:00", None, "2025-02-05 12:00:00", None,
            None, "2025-03-11 10:00:00", None, "2025-04-02 10:00:00", None, "2025-04-16 12:00:00"
        ],
        "bank": ["SBI", "HDFC", "SBI", "PNB", "ICICI", "SBI", "HDFC", "ICICI", "PNB", "SBI"],
        "bank_type": ["GOVERNMENT", "PRIVATE", "GOVERNMENT", "GOVERNMENT", "PRIVATE",
                      "GOVERNMENT", "PRIVATE", "PRIVATE", "GOVERNMENT", "GOVERNMENT"],
        "is_empty_text": [False] * 10,
        "source_file": ["sbi.csv", "hdfc.csv", "sbi.csv", "pnb.csv", "icici.csv",
                        "sbi.csv", "hdfc.csv", "icici.csv", "pnb.csv", "sbi.csv"],
        "source_folder": ["Govt_Banks", "Pvt_Banks", "Govt_Banks", "Govt_Banks", "Pvt_Banks",
                          "Govt_Banks", "Pvt_Banks", "Pvt_Banks", "Govt_Banks", "Govt_Banks"],
        "source_row": list(range(1, 11)),
    })
    # Run full NLP pipeline steps
    df = preprocessing.add_text_fields(df)
    df = language_detector.detect_language_frame(df)
    df = sentiment_engine.score_sentiment_frame(df)
    df = failure_detector.detect_failure_frame(df)
    df = risk_engine.classify_risk_frame(df)
    df = emotion_engine.detect_emotion_frame(df)
    df = aspect_engine.detect_aspects_frame(df)
    df = response_engine.add_response_features(df)
    df = sentiment_engine.rating_sentiment_mismatch(df)
    return df


def test_benjamini_hochberg():
    pvals = [0.01, 0.04, 0.03, 0.005, np.nan]
    qvals = statistical_analysis.benjamini_hochberg(pvals)
    assert len(qvals) == 5
    assert np.isnan(qvals[4])
    assert qvals[3] <= qvals[0] <= qvals[2] <= qvals[1]


def test_govt_vs_private_table(sample_master_df):
    table = statistical_analysis.govt_vs_private_table(sample_master_df)
    assert not table.empty
    assert "metric" in table.columns
    assert "government" in table.columns
    assert "private" in table.columns
    assert "q_value_bh" in table.columns


def test_temporal_monthly_trends(sample_master_df):
    table = temporal_engine.monthly_trends(sample_master_df)
    assert not table.empty
    assert "bank_type" in table.columns
    assert "month" in table.columns
    assert "negative_rate" in table.columns


def test_bank_improvement_matrix(sample_master_df):
    matrix = temporal_engine.bank_improvement_matrix(sample_master_df)
    assert not matrix.empty
    assert "bank" in matrix.columns
    assert "issue" in matrix.columns
    assert "difference" in matrix.columns


def test_response_summary_table(sample_master_df):
    table = response_engine.response_summary_table(sample_master_df)
    assert not table.empty
    assert "response_bucket" in table.columns
    assert "reviews" in table.columns


def test_pagerank_computation(sample_master_df):
    G = pagerank_engine.build_cooccurrence(sample_master_df)
    assert G.number_of_nodes() > 0


def test_data_validator_report(sample_master_df):
    report = data_validator.validate_master(sample_master_df)
    assert report["total_rows"] == 10
    assert report["empty_text_rows"] == 0
    assert "rating_distribution" in report


def test_annotation_sample_building(sample_master_df, tmp_path):
    """Builder must write where told and never touch the real annotations dir."""
    real = sc.PROJECT_ROOT / "data" / "annotations" / "annotation_sample.csv"
    before = real.stat().st_mtime if real.exists() else None

    sample = annotation.build_annotation_sample(
        sample_master_df, target=5, out_dir=tmp_path, force=True
    )
    assert not sample.empty
    assert "annotator_id" in sample.columns
    assert (tmp_path / "annotation_sample.csv").exists()
    assert (tmp_path / "annotation_sample_meta.json").exists()

    after = real.stat().st_mtime if real.exists() else None
    assert before == after, "real annotation sample was modified by the test!"


def test_annotation_overwrite_guard_warns(sample_master_df, tmp_path, caplog):
    """Replacing a larger existing sample must emit a loud OVERWRITE GUARD."""
    big = tmp_path / "annotation_sample.csv"
    big.write_text("a,b\n" + "\n".join("1,2" for _ in range(50)) + "\n")

    with caplog.at_level("WARNING"):
        annotation.build_annotation_sample(
            sample_master_df, target=5, out_dir=tmp_path
        )
    assert "OVERWRITE GUARD" in caplog.text


def test_annotation_default_path_is_sandboxed(sample_master_df):
    """During pytest the default output path must be a tmp dir, never the repo."""
    import os

    real = sc.PROJECT_ROOT / "data" / "annotations" / "annotation_sample.csv"
    before = os.stat(real).st_mtime_ns if real.exists() else None

    annotation.build_annotation_sample(sample_master_df, target=5, force=True)

    sandbox = annotation.sc.get_paths()["annotations_dir"]
    assert (sandbox / "annotation_sample.csv").exists()
    assert sc.PROJECT_ROOT not in sandbox.parents
    after = os.stat(real).st_mtime_ns if real.exists() else None
    assert before == after, "pytest modified the real annotation sample!"


def test_model_registry_device_and_helpers():
    device = model_registry.get_device()
    assert device in ("cpu", "mps", "cuda")
    slug = model_registry.slugify("distilbert/distilbert-base-uncased")
    assert "/" not in slug
