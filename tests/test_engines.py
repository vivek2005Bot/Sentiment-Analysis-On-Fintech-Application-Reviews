"""Unit tests for NLP engines: language, sentiment, failure, risk, emotion,
aspects, severity - using plan-specified example phrases (sections 9-16)."""
from __future__ import annotations

import pandas as pd

from src import (aspect_engine, emotion_engine, failure_detector,
                 language_detector, risk_engine, sentiment_engine)


def _frame(texts, ratings=None):
    return pd.DataFrame({
        "review_text": texts,
        "review_text_model": texts,
        "rating": ratings or [3] * len(texts),
        "failure_flag": ["NO"] * len(texts),
        "critical_failure": [False] * len(texts),
        "operational_friction": ["NO"] * len(texts),
        "friction_signals": [""] * len(texts),
    })


# --- language ---------------------------------------------------------------
def test_hinglish_detection():
    res = language_detector.detect_single("upi payment nahi ho raha")
    assert res["language"] == "Hinglish"


def test_native_hindi_script():
    res = language_detector.detect_single("यह ऐप बहुत अच्छा है")
    assert res["language"] == "Hindi"
    assert res["script"] == "Devanagari"


def test_english_detection():
    res = language_detector.detect_single(
        "The app works fine and transfers are quick. I use it daily.")
    assert res["language"] == "English"


def test_tanglish_detection():
    res = language_detector.detect_single("app romba slow ah irukku")
    assert res["language"] in ("Tanglish", "Hinglish", "Other_Indian")


def test_empty_text_nulls():
    res = language_detector.detect_single(None)
    assert res["language"] is None


# --- failure gate -----------------------------------------------------------
def test_crash_is_failure():
    df = failure_detector.detect_failure_frame(
        _frame(["app crashes every time I open it"]))
    assert df.loc[0, "failure_flag"] == "YES"


def test_friction_is_not_failure():
    df = failure_detector.detect_failure_frame(
        _frame(["The UI is confusing and slow, please improve navigation"]))
    assert df.loc[0, "failure_flag"] == "NO"
    assert df.loc[0, "operational_friction"] == "YES"


def test_money_deducted_is_critical_failure():
    df = failure_detector.detect_failure_frame(
        _frame(["money was deducted but not credited, no response"]))
    assert df.loc[0, "failure_flag"] == "YES"
    assert bool(df.loc[0, "critical_failure"]) is True


# --- risk -------------------------------------------------------------------
def test_financial_priority_rule():
    # mixed case: financial wins primary, technical becomes secondary (s.14)
    df = _frame(["app crashes when paying rent and my money is stuck"], [1])
    df["failure_flag"] = "YES"
    df["critical_failure"] = True
    out = risk_engine.classify_risk_frame(df)
    assert out.loc[0, "primary_risk"] == "FINANCIAL_SECURITY_RISK"
    assert out.loc[0, "secondary_risk"] == "TECHNICAL_PERFORMANCE_RISK"
    assert out.loc[0, "risk_overlap"] == "YES"
    assert out.loc[0, "severity"] == 5


def test_technical_risk():
    df = _frame(["app crashes every time, server down"], [1])
    df["failure_flag"] = "YES"
    out = risk_engine.classify_risk_frame(df)
    assert out.loc[0, "primary_risk"] == "TECHNICAL_PERFORMANCE_RISK"
    assert out.loc[0, "severity"] in (3, 4)


def test_non_failure_has_null_risk():
    out = risk_engine.classify_risk_frame(_frame(["nice app, works well"]))
    assert pd.isna(out.loc[0, "primary_risk"])


# --- sentiment --------------------------------------------------------------
def test_sentiment_positive_and_negative():
    df = sentiment_engine.score_sentiment_frame(
        _frame(["excellent app, very useful", "worst app ever, useless"],
               [5, 1]))
    assert df.loc[0, "sentiment"] == "POSITIVE"
    assert df.loc[1, "sentiment"] == "NEGATIVE"


def test_mismatch_flag():
    df = sentiment_engine.score_sentiment_frame(
        _frame(["worst experience, nothing works"], [5]))
    df = sentiment_engine.rating_sentiment_mismatch(df)
    assert bool(df.loc[0, "rating_sentiment_mismatch"]) is True


# --- emotion ----------------------------------------------------------------
def test_emotion_anger_with_emoji():
    df = emotion_engine.detect_emotion_frame(
        _frame(["Money deducted and nobody is helping 😡"]))
    assert df.loc[0, "emotion"] in ("ANGER", "FRUSTRATION", "FEAR")


def test_emotion_joy():
    df = emotion_engine.detect_emotion_frame(
        _frame(["happy with the smooth experience, great app"]))
    assert df.loc[0, "emotion"] == "JOY"


# --- aspects ----------------------------------------------------------------
def test_aspect_extraction_multi():
    res = aspect_engine.extract_aspects_single(
        "App is good but UPI is terrible and customer service is useless.")
    assert "UPI" in (res["aspects"] or "")
    assert "CUSTOMER_SERVICE" in (res["aspects"] or "")
    assert res["aspect_count"] >= 2


def test_aspect_sentiment_direction():
    res = aspect_engine.extract_aspects_single(
        "App is good but UPI is terrible and customer service is useless.")
    sents = dict(pair.split(":") for pair in
                 res["aspect_sentiment"].split(";") if ":" in pair)
    assert sents.get("UPI") == "NEGATIVE"
    assert sents.get("APP_PERFORMANCE", sents.get("UI_UX", "POSITIVE")) in \
        ("POSITIVE", "NEUTRAL")


def test_aspect_empty():
    res = aspect_engine.extract_aspects_single(None)
    assert res["aspects"] is None
