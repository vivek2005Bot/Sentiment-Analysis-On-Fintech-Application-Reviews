"""Baseline sentiment engine v0 (deterministic rules).

IMPORTANT (plan Rules 8, 9): this lexicon+rating-prior scorer is a
*baseline/bootstrap* signal only. It is never presented as the final
classifier. Trained transformers replace it (see model_registry / train.py);
predictions record their `model_source` so the dashboard is always honest
about which method produced each number.
"""

from __future__ import annotations

import pandas as pd

import src.config as sc
from src.lexicons import NEG_RE, POS_RE
from src.utils import setup_logging

logger = setup_logging("sentiment_engine")

VERSION = "rules_lexicon_rating_v1"


def score_sentiment_frame(
    df: pd.DataFrame, text_col: str = "review_text_model"
) -> pd.DataFrame:
    """Add sentiment + confidence using lexical score blended with rating prior."""
    cfg = sc.get_config()["sentiment_rules"]
    out = df.copy()
    text = out[text_col].astype("string").fillna("")
    pos = text.str.count(POS_RE).astype("float64")
    neg = text.str.count(NEG_RE).astype("float64")
    lex = (pos - neg) / (pos + neg + 1.0)

    rating = pd.to_numeric(out.get("rating"), errors="coerce")
    prior = rating.map(cfg["rating_prior"]).astype("float64")
    w = float(cfg["rating_prior_weight"])
    score = (1.0 - w) * lex + w * prior.fillna(0.0)

    labels = pd.Series("NEUTRAL", index=out.index, dtype="object")
    labels[score > 0.15] = "POSITIVE"
    labels[score < -0.15] = "NEGATIVE"

    has_rating = prior.notna()
    confidence = pd.Series(0.5, index=out.index, dtype="float64")
    confidence.loc[has_rating] = 0.55 + 0.4 * score.loc[has_rating].abs()
    confidence.loc[~has_rating] = 0.5 + 0.4 * score.loc[~has_rating].abs()
    confidence = confidence.clip(upper=0.97)

    out["sentiment_score"] = score.round(4)
    out["sentiment"] = labels
    out["sentiment_confidence"] = confidence.round(3)
    out["sentiment_model_source"] = VERSION
    logger.info("Sentiment (rules): %s", out["sentiment"].value_counts().to_dict())
    return out


def rating_sentiment_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rating/text disagreements for investigation (plan section 29).

    Mismatch is NOT automatically sarcasm - it is a review flag for
    human inspection.
    """
    out = df.copy()
    rating = pd.to_numeric(out.get("rating"), errors="coerce")
    mismatch = ((rating >= 4) & (out["sentiment"] == "NEGATIVE")) | (
        (rating <= 2) & (out["sentiment"] == "POSITIVE")
    )
    out["rating_sentiment_mismatch"] = mismatch.fillna(False)
    return out
