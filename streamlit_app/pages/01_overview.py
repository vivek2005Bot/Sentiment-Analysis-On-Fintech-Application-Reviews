"""Page: Executive Overview (plan s.49)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import (bar_chart, donut_chart, figure, get_filtered,
                               kpi_cards, line_chart, section_header,
                               sentiment_bar_chart)


def render() -> None:
    st.markdown("""
    <h1 style="font-weight: 800; letter-spacing: -0.02em;">
        🏦 Executive Overview
    </h1>
    <p style="opacity: 0.55; font-size: 0.85rem; margin-top: -8px;">
        Decision-support overview over precomputed engine outputs.
        KPIs state their denominators (plan s.58).
    </p>
    """, unsafe_allow_html=True)
    df = get_filtered()
    kpi_cards(df)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        section_header("📊", "Sentiment Distribution")
        sentiment_bar_chart(df["sentiment"].value_counts(),
                            title="Sentiment Distribution")
    with c2:
        section_header("🎯", "Risk Distribution",
                       "Failure rows only (Tier-2 risk)")
        risk = df["primary_risk"].value_counts()
        if not risk.empty:
            donut_chart(risk, title="Risk Categories")
        else:
            bar_chart(pd.Series({"no failure signal": len(df)}),
                      title="Risk Distribution")
        st.caption(f"{int((df['failure_flag'] == 'YES').sum()):,} of "
                   f"{len(df):,} reviews gate into Tier-2 risk.")

    st.divider()
    section_header("⚖️", "Government vs Private Negative Rate",
                   "Denominator: all reviews within each bank type.")
    gvp = (df.groupby("bank_type")["sentiment"]
           .apply(lambda s: (s == "NEGATIVE").mean() * 100).round(2))
    bar_chart(gvp, y_title="Negative rate (%)")

    section_header("📈", "Reviews Over Time")
    d = df.dropna(subset=["review_date"]).copy()
    if d.empty:
        st.info("No dated reviews in the current filter.")
    else:
        d["month"] = (pd.to_datetime(d["review_date"])
                      .dt.to_period("M").dt.to_timestamp())
        line_chart(d.groupby("month").size(), y_title="Reviews")

    st.divider()
    section_header("🏆", "Bank League Table")
    summary = (df.groupby(["bank", "bank_type"], observed=True).agg(
        reviews=("review_id", "size"),
        mean_rating=("rating", "mean"),
        negative_rate=("sentiment", lambda s: (s == "NEGATIVE").mean()),
        failure_rate=("failure_flag", lambda s: (s == "YES").mean()),
        financial_risk=("primary_risk",
                        lambda s: (s == "FINANCIAL_SECURITY_RISK").mean()),
        severity=("severity", "mean"),
    ).reset_index().round(4).sort_values("reviews", ascending=False))
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    section_header("🧩", "Top Aspects Overall")
    counts = (df["aspects"].dropna().str.split(",").explode()
              .value_counts().head(10))
    if not counts.empty:
        bar_chart(counts, horizontal=True, y_title="Mentions")

    st.divider()
    section_header("🗺️", "Dashboard Navigation")
    nav_cols = st.columns(3)
    pages_info = [
        ("🙂", "Sentiment Intelligence", "Sentiment by bank/rating/language/time"),
        ("⚠️", "Risk Intelligence", "Tier-1 gate + Tier-2 risk breakdown"),
        ("⚖️", "Government vs Private", "Statistical comparison with effect sizes"),
        ("🌐", "Language & Hinglish", "Multilingual/code-mixed coverage"),
        ("🤖", "Model Benchmark", "Held-out transformer results and baselines"),
        ("🧩", "Aspect & PageRank", "Aspect taxonomy, co-occurrence centrality"),
        ("💬", "Response Intelligence", "Developer response latency analysis"),
        ("🧭", "Issue Explorer", "Term-level issue intelligence"),
        ("🔍", "Review Investigator", "Keyword search over individual reviews"),
        ("🚨", "Critical Queue", "Prioritised human-action worklist"),
        ("🔎", "Model Explainability", "Token attribution for trained models"),
        ("📈", "Early Warning", "Anomaly signals vs rolling baselines"),
        ("📚", "Data & Methodology", "Sources, label definitions, limitations"),
    ]
    for i, (icon, name, desc) in enumerate(pages_info):
        nav_cols[i % 3].markdown(f"**{icon} {name}**  \n{desc}")


if __name__ == "__main__":
    render()
else:
    render()
