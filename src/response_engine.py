"""Response engine (plan section 35): developer response latency analysis.

Only meaningful when reply_text + reply_date exist (Govt + Pvt files do).
Computes response_time_hours, latency buckets, and (optionally, when
follow-up reviews exist) tests latency vs rating/sentiment change with
observational framing: 'associated with', never causal language.
"""

from __future__ import annotations

import pandas as pd

import src.config as sc
from src.utils import setup_logging

logger = setup_logging("response_engine")


def add_response_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = sc.get_config()["response"]
    out = df.copy()
    rd = pd.to_datetime(out.get("review_date"), errors="coerce")
    pd_ = pd.to_datetime(out.get("reply_date"), errors="coerce")
    has_reply = out.get("reply_text", pd.Series(pd.NA, index=out.index)).notna()

    hours = (pd_ - rd).dt.total_seconds() / 3600.0
    out["response_time_hours"] = hours.where(has_reply)
    neg = out["response_time_hours"] < 0  # data quirk: reply before review
    out.loc[neg, "response_time_hours"] = pd.NA

    edges = cfg["buckets_hours"]
    labels = cfg["bucket_labels"]

    def _to_bucket(h: float) -> str:
        for edge, lab in zip(edges, labels):
            if h < edge:
                return lab
        return labels[-1]

    bucket = pd.Series(pd.NA, index=out.index, dtype="object")
    valid = out["response_time_hours"].notna()
    if valid.any():
        bucket.loc[valid] = out.loc[valid, "response_time_hours"].map(_to_bucket)
    bucket.loc[~valid] = "No response"
    bucket.loc[neg] = pd.NA  # anomalous latency excluded from buckets
    out["response_bucket"] = bucket

    out["has_developer_response"] = has_reply
    latency = out["response_time_hours"].dropna()
    logger.info(
        "Responses: %d | median latency: %.1f h",
        int(latency.shape[0]),
        float(latency.median()) if len(latency) else -1,
    )
    return out


def response_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 7 (plan s.55): response bucket vs rating/sentiment stats."""
    out = df.copy()
    rows = []
    for bucket, sub in out.groupby("response_bucket", dropna=False, observed=True):
        rows.append(
            {
                "response_bucket": bucket if pd.notna(bucket) else "Unknown",
                "reviews": int(len(sub)),
                "mean_rating": (
                    round(float(sub["rating"].mean()), 3)
                    if sub["rating"].notna().any()
                    else None
                ),
                "negative_rate": round(
                    float((sub["sentiment"] == "NEGATIVE").mean()), 4
                ),
                "failure_rate": round(float((sub["failure_flag"] == "YES").mean()), 4),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("reviews", ascending=False)


def latency_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """Association of latency with rating (Mann-Whitney U, observational)."""
    from scipy.stats import mannwhitneyu

    out = df.copy()
    fast = out.loc[out["response_bucket"] == "<24h", "rating"].dropna()
    slow = out.loc[out["response_bucket"] == ">72h", "rating"].dropna()
    rows = [
        {
            "comparison": "rating:<24h vs >72h",
            "n_fast": int(len(fast)),
            "n_slow": int(len(slow)),
            "median_fast": float(fast.median()) if len(fast) else None,
            "median_slow": float(slow.median()) if len(slow) else None,
        }
    ]
    if len(fast) > 20 and len(slow) > 20:
        stat, p = mannwhitneyu(fast, slow, alternative="two-sided")
        rows[0].update(u_statistic=float(stat), p_value=float(p))
    return pd.DataFrame(rows)
