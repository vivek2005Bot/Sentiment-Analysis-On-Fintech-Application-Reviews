"""Tier-1 failure gate (plan sections 12-13): failure vs operational friction.

Deterministic rule engine v1. A negative review is NOT automatically a
failure: friction (slow UX, feature requests) stays separate from failure
(system/transaction problems). Hinglish phrases are included in the
patterns (plan section 9/12). Outputs YES/NO strings per plan schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lexicons import FAILURE_RES, FRICTION_RES
from src.utils import setup_logging

logger = setup_logging("failure_detector")

VERSION = "rules_lexicon_v1"


def _match_table(text: pd.Series, res: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {name: text.str.contains(rex, na=False) for name, rex in res.items()},
        index=text.index,
    )


def detect_failure_frame(
    df: pd.DataFrame, text_col: str = "review_text_model"
) -> pd.DataFrame:
    """Add failure_flag, operational_friction, critical_failure + signals."""
    out = df.copy()
    text = out[text_col].astype("string").fillna("")
    fail_hits = _match_table(text, FAILURE_RES)
    fric_hits = _match_table(text, FRICTION_RES)

    failure_any = fail_hits.any(axis=1)
    friction_any = fric_hits.any(axis=1)
    n_fail_cats = fail_hits.sum(axis=1)

    critical = fail_hits["transaction_failure"] | fail_hits["unauthorized"]

    out["failure_flag"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[failure_any, "failure_flag"] = "YES"
    out.loc[~failure_any, "failure_flag"] = "NO"
    out["operational_friction"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[friction_any, "operational_friction"] = "YES"
    out.loc[~friction_any, "operational_friction"] = "NO"
    out["critical_failure"] = critical

    conf = (0.55 + 0.07 * n_fail_cats).clip(upper=0.95)
    out["failure_confidence"] = pd.Series(np.nan, index=out.index, dtype="float64")
    out.loc[failure_any, "failure_confidence"] = conf.loc[failure_any]

    out["failure_signals"] = fail_hits.apply(
        lambda row: ";".join([k for k, v in row.items() if v]), axis=1
    )
    out["friction_signals"] = fric_hits.apply(
        lambda row: ";".join([k for k, v in row.items() if v]), axis=1
    )
    out["failure_model_source"] = VERSION

    logger.info(
        "Failure gate: YES=%d NO=%d | friction YES=%d | critical=%d",
        int((out["failure_flag"] == "YES").sum()),
        int((out["failure_flag"] == "NO").sum()),
        int((out["operational_friction"] == "YES").sum()),
        int(out["critical_failure"].sum()),
    )
    return out
