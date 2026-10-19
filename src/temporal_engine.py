"""Temporal analysis + early warning (plan sections 37, 52, 51, 56).

Monthly trends for sentiment/failure/risk metrics; anomaly detection vs
rolling baselines (analytical alert only - never proof of an outage);
industry-level bank improvement matrix from OBSERVED issue categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as sc
from src.utils import save_json, setup_logging

logger = setup_logging("temporal_engine")

# Observed issue category -> suggested operational area (plan s.51)
OPERATIONAL_AREA = {
    "OTP": "OTP delivery and verification",
    "LOGIN": "Login and credential flows",
    "UPI": "UPI payment processing",
    "NEFT": "NEFT transfers",
    "IMPS": "IMPS transfers",
    "MONEY_TRANSFER": "Fund transfer journeys",
    "TRANSACTION": "Transaction processing",
    "SERVER": "Server capacity and uptime",
    "CRASH": "App stability",
    "APP_PERFORMANCE": "App performance",
    "ATM": "ATM/card services",
    "CUSTOMER_SERVICE": "Customer service responsiveness",
    "FRAUD": "Fraud monitoring",
    "SECURITY": "Security posture",
    "ACCOUNT": "Account services",
    "KYC": "KYC journeys",
    "CREDIT_CARD": "Credit card services",
    "LOAN": "Loan servicing",
    "CHARGES": "Fee transparency",
    "NOTIFICATION": "Notification reliability",
    "UI_UX": "Interface usability",
    "FEATURE_REQUEST": "Feature backlog",
}


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = (
        pd.to_datetime(out["review_date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    return out


RATE_FNS = {
    "negative_rate": lambda s: (s["sentiment"] == "NEGATIVE").mean(),
    "failure_rate": lambda s: (s["failure_flag"] == "YES").mean(),
    "technical_risk_rate": lambda s: (
        s["primary_risk"] == "TECHNICAL_PERFORMANCE_RISK"
    ).mean(),
    "financial_risk_rate": lambda s: (
        s["primary_risk"] == "FINANCIAL_SECURITY_RISK"
    ).mean(),
    "friction_rate": lambda s: (s["operational_friction"] == "YES").mean(),
}


def monthly_trends(master: pd.DataFrame) -> pd.DataFrame:
    df = add_month(master)
    rows = []
    for (group, month), sub in df.groupby(
        ["bank_type", "month"], dropna=False, observed=True
    ):
        if pd.isna(month):
            continue
        row = {"bank_type": group, "month": month, "reviews": len(sub)}
        for metric, fn in RATE_FNS.items():
            row[metric] = round(float(fn(sub)), 5)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["bank_type", "month"])
    table.to_csv(sc.get_paths()["tables_dir"] / "temporal_trends.csv", index=False)
    return table


def early_warning(master: pd.DataFrame) -> pd.DataFrame:
    """Compare last full month vs trailing baseline (z-score alerting)."""
    cfg = sc.get_config()["early_warning"]
    df = add_month(master)
    last_month = df["month"].max()
    baseline_months = pd.date_range(
        end=last_month, periods=cfg["baseline_months"] + 1, freq="MS"
    )[:-1]
    rows = []
    for group, sub in df.groupby("bank_type", observed=True):
        base = sub[sub["month"].isin(baseline_months)]
        cur = sub[sub["month"] == last_month]
        if len(base) < cfg["min_baseline_reviews"] or len(cur) < 50:
            continue
        for metric, fn in RATE_FNS.items():
            if metric == "friction_rate":
                continue
            hist = [float(fn(sub[sub["month"] == m])) for m in baseline_months]
            mean, std = float(np.mean(hist)), float(np.std(hist))
            cur_rate = float(fn(cur))
            z = (cur_rate - mean) / std if std > 1e-9 else 0.0
            rows.append(
                {
                    "bank_type": group,
                    "metric": metric,
                    "baseline_months": cfg["baseline_months"],
                    "baseline_rate": round(mean, 5),
                    "current_month": str(last_month.date()),
                    "current_rate": round(cur_rate, 5),
                    "z_score": round(z, 2),
                    "status": (
                        "Potential anomaly" if z >= cfg["z_threshold"] else "Normal"
                    ),
                    "note": "Analytical early-warning signal, not an outage "
                    "confirmation.",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(sc.get_paths()["tables_dir"] / "early_warning.csv", index=False)
    flags = table[table["status"] == "Potential anomaly"]
    save_json(
        flags.to_dict(orient="records"),
        sc.get_paths()["tables_dir"] / "potential_anomalies.json",
    )
    return table


def bank_improvement_matrix(master: pd.DataFrame) -> pd.DataFrame:
    """Industry-level improvement matrix (plan s.51).

    bank_rate vs industry rate (rest of corpus) on OBSERVED issue categories;
    suggested_operational_area maps the observed category only - it never
    invents internal knowledge about the bank.
    """
    rows = []
    for bank, sub in master.groupby("bank", observed=True):
        others = master[master["bank"] != bank]
        for metric, fn in RATE_FNS.items():
            if metric == "friction_rate":
                continue
            b_rate, o_rate = float(fn(sub)), float(fn(others))
            rows.append(
                {
                    "bank": bank,
                    "issue": metric,
                    "bank_rate": round(b_rate, 4),
                    "industry_rate": round(o_rate, 4),
                    "difference": round(b_rate - o_rate, 4),
                    "volume": int(len(sub)),
                    "avg_severity": round(float(sub["severity"].mean()), 2),
                    "suggested_operational_area": OPERATIONAL_AREA.get(
                        metric.replace("_rate", "").upper(), "Observed issue"
                    ),
                }
            )
        flagged = sub[sub["aspects"].notna()]
        for aspect, asub in flagged.groupby("aspects", observed=True):
            aspect = str(aspect)
            if "," in aspect:
                continue  # single labels only for the matrix
            others_a = master[(master["bank"] != bank) & (master["aspects"] == aspect)]
            if others_a.empty or len(asub) < 20:
                continue
            b = float((asub["sentiment"] == "NEGATIVE").mean())
            o = float((others_a["sentiment"] == "NEGATIVE").mean())
            rows.append(
                {
                    "bank": bank,
                    "issue": f"aspect:{aspect}",
                    "bank_rate": round(b, 4),
                    "industry_rate": round(o, 4),
                    "difference": round(b - o, 4),
                    "volume": int(len(asub)),
                    "avg_severity": round(float(asub["severity"].mean()), 2),
                    "suggested_operational_area": OPERATIONAL_AREA.get(
                        aspect, "Observed issue"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        sc.get_paths()["tables_dir"] / "bank_improvement_matrix.csv", index=False
    )
    return table
