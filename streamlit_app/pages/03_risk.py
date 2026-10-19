"""Page: Risk Intelligence (plan s.49) - the two-tier risk architecture."""
from __future__ import annotations

import streamlit as st

from components.common import (bar_chart, donut_chart, figure, get_filtered,
                               kpi_cards, load_table, missing_table_notice,
                               review_list, section_header)


def render() -> None:
    st.markdown("""
    <h1 style="font-weight: 800; letter-spacing: -0.02em;">
        ⚠️ Risk Intelligence
    </h1>
    <p style="opacity: 0.55; font-size: 0.85rem; margin-top: -8px;">
        Tier 1 = failure gate; Tier 2 = primary risk
        (mutually exclusive, financial wins overlap). Risk is null
        for failure=NO rows, so risk denominators exclude them.
    </p>
    """, unsafe_allow_html=True)
    df = get_filtered()
    kpi_cards(df)
    st.divider()

    failures = df[df["failure_flag"] == "YES"]
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Failure Reviews", f"{len(failures):,}")
    c2.metric("⚙️ Technical Risk",
              f"{(failures['primary_risk'] == 'TECHNICAL_PERFORMANCE_RISK').sum():,}",
              help="Share: "
                   f"{(failures['primary_risk'] == 'TECHNICAL_PERFORMANCE_RISK').mean() * 100:.1f}% "
                   "of failure reviews")
    c3.metric("💰 Financial Risk",
              f"{(failures['primary_risk'] == 'FINANCIAL_SECURITY_RISK').sum():,}",
              help="Share: "
                   f"{(failures['primary_risk'] == 'FINANCIAL_SECURITY_RISK').mean() * 100:.1f}% "
                   "of failure reviews")

    st.divider()
    section_header("🏦", "Risk by Bank")
    t = load_table("risk_by_bank.csv")
    if t.empty:
        missing_table_notice("risk_by_bank.csv")
    else:
        st.dataframe(t, use_container_width=True, hide_index=True)
        st.caption("Rates use ALL of that bank's reviews as the denominator.")

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        section_header("🔀", "Mixed-risk Overlap (plan s.14)")
        overlap = failures[failures["risk_overlap"] == "YES"]
        st.caption(f"{len(overlap):,} failure reviews fire both signal families; "
                   "primary follows the financial-first priority rule and the "
                   "secondary signal is retained.")
        if not overlap.empty:
            st.dataframe(
                overlap[["bank", "primary_risk", "secondary_risk", "review_text"]]
                .head(20), use_container_width=True, hide_index=True)
    with r2:
        section_header("🔍", "Failure Signal Categories")
        signals = (df["failure_signals"].dropna().str.split(";").explode()
                   .replace("", None).dropna().value_counts())
        if not signals.empty:
            donut_chart(signals.head(8), title="Failure Signal Types")

    st.divider()
    section_header("📊", "Severity Distribution")
    bar_chart(df["severity"].value_counts().sort_index(), y_title="Reviews")
    st.caption("Severity is a project-defined annotation policy (s.16), not a "
               "universal risk score.")

    st.divider()
    section_header("🖼️", "Precomputed Risk Figures")
    fc1, fc2 = st.columns(2)
    with fc1:
        figure("04_technical_vs_financial_risk.png")
        figure("07_risk_by_rating.png")
    with fc2:
        figure("05_risk_by_bank.png")
        figure("20_severity_distribution.png")

    st.divider()
    section_header("📋", "Sample Failure Reviews",
                   "Highest-severity failures")
    review_list(failures.sort_values("severity", ascending=False), limit=10,
                caption="Highest-severity failures")


if __name__ == "__main__":
    render()
else:
    render()
