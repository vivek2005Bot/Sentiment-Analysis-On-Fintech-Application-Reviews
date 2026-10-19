"""Page: Sentiment Intelligence (plan s.49)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import (bar_chart, donut_chart, figure, get_filtered,
                               grouped_bar_chart, kpi_cards, line_chart,
                               load_table, section_header,
                               sentiment_bar_chart)


def render() -> None:
    st.markdown("""
    <h1 style="font-weight: 800; letter-spacing: -0.02em;">
        🙂 Sentiment Intelligence
    </h1>
    """, unsafe_allow_html=True)
    df = get_filtered()
    kpi_cards(df)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        section_header("🏦", "Sentiment by Bank (%)")
        g = (df.groupby("bank")["sentiment"].value_counts(normalize=True)
             .unstack(fill_value=0).mul(100).round(2))
        st.dataframe(g, use_container_width=True)
    with c2:
        section_header("⭐", "Sentiment by Rating")
        g = (df.dropna(subset=["rating"]).groupby("rating")["sentiment"]
             .value_counts().unstack(fill_value=0))
        grouped_bar_chart(g, title="Sentiment by Rating", y_title="Reviews")

    st.divider()
    section_header("📈", "Sentiment Over Time")
    d = df.dropna(subset=["review_date"]).copy()
    if d.empty:
        st.info("No dated reviews in the current filter.")
    else:
        d["month"] = (pd.to_datetime(d["review_date"])
                      .dt.to_period("M").dt.to_timestamp())
        line_chart(d.groupby(["month", "sentiment"]).size().unstack(
            fill_value=0), y_title="Reviews")
        st.caption("Denominator: reviews with a parsed review_date.")

    section_header("🌐", "Sentiment by Language")
    st.dataframe(
        df.groupby("language")["sentiment"].value_counts()
        .unstack(fill_value=0).fillna(0).astype(int),
        use_container_width=True)

    with st.expander("⚠️ Rating/text disagreement (plan s.29 - not sarcasm)"):
        st.caption("Mismatch is flagged for human review; it is NOT labelled "
                   "as sarcasm by the pipeline.")
        mism = df[df["rating_sentiment_mismatch"] == True]  # noqa: E712
        st.metric("Mismatched reviews", f"{len(mism):,}",
                  help="4-5★ with NEGATIVE text, or 1-2★ with POSITIVE text")
        st.dataframe(mism[["bank", "rating", "sentiment", "review_text"]]
                     .head(50), use_container_width=True, hide_index=True)

    st.divider()
    section_header("🖼️", "Precomputed Figures (s.56)")
    fig_c1, fig_c2 = st.columns(2)
    with fig_c1:
        figure("01_sentiment_distribution.png")
    with fig_c2:
        figure("02_sentiment_over_time.png")

    mism_path = load_table("sentiment_rating_mismatch.csv")
    if not mism_path.empty:
        section_header("⚡", "Mismatch Summary")
        st.dataframe(mism_path, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
else:
    render()
