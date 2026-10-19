"""Emotion engine (plan section 15): expressed emotional state.

Single primary emotion + multi-label inventory (`emotion_all`), deterministic
lexicon v1 with emoji cues preserved as signal (plan section 28).
"""

from __future__ import annotations

import pandas as pd

from src.lexicons import EMOTION_RES
from src.utils import setup_logging

logger = setup_logging("emotion_engine")

VERSION = "lexicon_v1"

# Priority order for the single primary label (plan: multi-label optional).
PRIORITY = [
    "FEAR",
    "ANGER",
    "FRUSTRATION",
    "DISAPPOINTMENT",
    "SADNESS",
    "JOY",
    "TRUST",
    "CONFUSION",
]


def detect_emotion_frame(
    df: pd.DataFrame, text_col: str = "review_text_model"
) -> pd.DataFrame:
    out = df.copy()
    text = out[text_col].astype("string").fillna("")
    hits = pd.DataFrame(
        {name: text.str.contains(rex, na=False) for name, rex in EMOTION_RES.items()},
        index=out.index,
    )

    primary = pd.Series("NEUTRAL", index=out.index, dtype="object")
    remaining = pd.Series(True, index=out.index)
    for emo in PRIORITY:
        take = hits[emo] & remaining
        primary[take] = emo
        remaining &= ~take

    all_labels = hits.apply(
        lambda row: ",".join([k for k in PRIORITY if row[k]]), axis=1
    )
    n = hits.sum(axis=1)
    conf = (0.55 + 0.06 * n).clip(upper=0.90)
    conf[primary == "NEUTRAL"] = 0.50

    out["emotion"] = primary
    out["emotion_all"] = all_labels
    out["emotion_confidence"] = conf.round(3)
    out["emotion_model_source"] = VERSION
    logger.info("Emotion: %s", out["emotion"].value_counts().to_dict())
    return out
