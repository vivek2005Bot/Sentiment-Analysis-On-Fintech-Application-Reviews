"""Unit tests for data layer: column mapping, PII masking, removal log."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import _map_columns
from src.utils import RemovalLog, mask_pii


def test_column_mapping_google_play_schema():
    cols = ["reviewId", "userName", "userImage", "content", "score",
            "thumbsUpCount", "reviewCreatedVersion", "at", "replyContent",
            "repliedAt", "appVersion"]
    m = _map_columns(cols)
    assert m["review_id"] == "reviewId"
    assert m["review_text"] == "content"
    assert m["rating"] == "score"
    assert m["review_date"] == "at"
    assert m["reply_text"] == "replyContent"
    assert m["reply_date"] == "repliedAt"
    assert m["app_version"] == "appVersion"


def test_column_mapping_alternative_names():
    m = _map_columns(["review_text", "rating", "review_date", "review_id"])
    assert m["review_text"] == "review_text"
    assert m["rating"] == "rating"


def test_column_mapping_case_insensitive_and_missing():
    m = _map_columns(["CONTENT", "SCORE"])
    assert m["review_text"] == "CONTENT"
    assert m["rating"] == "SCORE"
    assert "review_date" not in m


def test_mask_pii_email():
    assert mask_pii("contact me at john.doe@gmail.com now") == \
        "contact me at j***@gmail.com now"


def test_mask_pii_phone():
    out = mask_pii("call 9876543210 for help")
    assert "9876543210" not in out
    assert "3210" in out


def test_mask_pii_account_number():
    out = mask_pii("my account 123456789012 is locked")
    assert "123456789012" not in out
    assert "****9012" in out


def test_mask_pii_empty_and_none():
    assert mask_pii(None) == ""
    assert mask_pii(float("nan")) == ""
    assert mask_pii("no pii here") == "no pii here"


def test_removal_log_math():
    rl = RemovalLog()
    rl.add("test_stage", "reason", before=100, removed=10, after=90,
           source="f.csv")
    df = rl.as_frame()
    assert df.loc[0, "rows_removed"] == 10
    assert df.loc[0, "rows_after"] == 90
    # consistency guard: wrong 'removed' corrected
    rl.add("s2", "r", before=50, removed=999, after=45)
    assert rl.as_frame().loc[1, "rows_removed"] == 5
