"""Tier-2 risk classifier (plan sections 13-14): TECHNICAL vs FINANCIAL.

Deterministic v1 with the plan's priority rule: financial/security signals
win the primary label when both signal families fire; the other family
becomes the secondary signal and `risk_overlap` records the ambiguity.
The official two-class benchmark uses ONLY primary_risk (mutually exclusive).
Also implements the severity policy (plan section 16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lexicons import FIN_RISK_RE, TECH_RISK_RE
from src.utils import setup_logging

logger = setup_logging("risk_engine")

VERSION = "priority_financial_first_v1"

TECH = "TECHNICAL_PERFORMANCE_RISK"
FIN = "FINANCIAL_SECURITY_RISK"

# Severity policy (plan section 16): 1 suggestion .. 5 financial/security.
# 5 financial | 4 critical technical | 3 other failure | 2 friction | 1 suggestion


def classify_risk_frame(
    df: pd.DataFrame, text_col: str = "review_text_model"
) -> pd.DataFrame:
    """Assign primary/secondary risk + severity to every review."""
    out = df.copy()
    text = out[text_col].astype("string").fillna("")
    tech = text.str.contains(TECH_RISK_RE, na=False)
    fin = text.str.contains(FIN_RISK_RE, na=False)
    failure = (
        out.get("failure_flag", pd.Series("NO", index=out.index)) == "YES"
    ).fillna(False)
    critical = out.get("critical_failure", pd.Series(False, index=out.index))
    critical = critical.fillna(False).astype(bool)

    primary = pd.Series(pd.NA, index=out.index, dtype="object")
    primary[fin] = FIN
    primary[~fin & tech] = TECH
    # non-failure reviews have no Tier-2 risk label (null, per plan s.40)
    primary[~failure] = pd.NA

    secondary = pd.Series(pd.NA, index=out.index, dtype="object")
    both = fin & tech
    is_fin = (primary == FIN).fillna(False)
    is_tech = (primary == TECH).fillna(False)
    secondary[both & is_fin] = TECH
    secondary[both & is_tech] = FIN

    n_sig = fin.astype(int) + tech.astype(int)
    conf = (0.60 + 0.08 * n_sig).clip(upper=0.95)
    risk_conf = pd.Series(np.nan, index=out.index, dtype="float64")
    risk_conf[failure] = conf.loc[failure]

    # severity: financial=5; critical technical=4; other failure=3;
    # friction=2 (1 for pure feature requests); otherwise 1.
    severity = pd.Series(1, index=out.index, dtype="int64")
    severity[failure & tech] = 4  # technical failure
    severity[failure & tech & ~critical] = 3  # non-critical technical failure
    severity[failure & critical] = 4  # critical failure
    severity[(primary == FIN).fillna(False)] = 5  # financial/security incident
    friction = (
        out.get("operational_friction", pd.Series("NO", index=out.index)) == "YES"
    ).fillna(False)
    severity[friction] = 2
    featurish = (
        out.get("friction_signals", pd.Series("", index=out.index))
        .fillna("")
        .str.contains("feature_request", na=False)
    )
    severity[friction & featurish & ~failure] = 1

    out["primary_risk"] = primary
    out["primary_risk_confidence"] = risk_conf
    out["secondary_risk"] = secondary
    out["risk_overlap"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[both & failure, "risk_overlap"] = "YES"
    out.loc[failure & ~both, "risk_overlap"] = "NO"
    out["severity"] = severity
    out["risk_model_source"] = VERSION

    counts = out["primary_risk"].value_counts(dropna=False).to_dict()
    logger.info(
        "Risk (failure rows only): %s | overlap YES=%d",
        counts,
        int((out["risk_overlap"] == "YES").sum()),
    )
    return out
