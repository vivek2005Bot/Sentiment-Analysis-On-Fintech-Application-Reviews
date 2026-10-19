"""Page: Early Warning (plan s.49, s.52, s.37)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import figure, get_filtered, line_chart, load_table


def render() -> None:
    st.title("📈 Early Warning")
    st.caption("Analytical anomaly detection against a rolling historical "
               "baseline. A flag is NOT confirmation of a production outage "
               "(plan s.52).")
    df = get_filtered()

    ew = load_table("early_warning.csv")
    if ew.empty:
        st.warning("`early_warning.csv` not found - run the full pipeline.")
    else:
        anomalies = ew[ew["status"] == "Potential anomaly"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Signals evaluated", len(ew))
        c2.metric("Potential anomalies", len(anomalies))
        c3.metric("Baseline window", f"{int(ew['baseline_months'].iloc[0])} months")

        for _, row in anomalies.iterrows():
            st.warning(
                f"**{row['bank_type']} · {row['metric']}** — "
                f"baseline {row['baseline_rate']:.3%} → current "
                f"{row['current_rate']:.3%} (z={row['z_score']}) "
                f"as of {row['current_month']}")

        st.subheader("All signals")
        st.dataframe(ew, width="stretch")
        st.caption("z-score = (current month rate - baseline mean) / baseline "
                   "standard deviation. Threshold and minimum baseline volume "
                   "come from configs/config.yaml (early_warning).")

    st.subheader("Trend context")
    trend = load_table("temporal_trends.csv")
    if trend.empty:
        st.warning("`temporal_trends.csv` not found.")
    else:
        metric = st.selectbox(
            "Metric", ["negative_rate", "failure_rate", "technical_risk_rate",
                       "financial_risk_rate", "friction_rate"])
        piv = trend.pivot_table(index="month", columns="bank_type",
                                values=metric)
        line_chart(piv, y_title="Rate")
        st.caption(f"Monthly {metric}; denominators are that month's reviews "
                   "within each bank type.")

        bank_type = st.selectbox("Detail for bank type",
                                 sorted(trend["bank_type"].unique()))
        st.dataframe(trend[trend["bank_type"] == bank_type],
                     width="stretch")

    st.divider()
    figure("22_potential_anomaly_timeline.png")
    figure("21_app_version_trend.png",
           "App-version trends are associations only - a release is not "
           "claimed to have caused any observed change.")


if __name__ == "__main__":
    render()
else:
    render()
