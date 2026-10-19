"""Page: Issue Explorer (plan s.50).

An analyst types a term (UPI, OTP, server, fraud, money deducted, login) and
gets mention counts, affected banks, Government/Private split, sentiment,
risk, emotion, trend, language distribution and representative reviews.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import (bar_chart, get_filtered, line_chart, pii,
                               review_list, sentiment_bar_chart)


def render() -> None:
    st.title("🧭 Issue Explorer")
    st.caption("Term-level issue intelligence across the corpus. All review "
               "text is PII-masked (plan s.50).")
    df = get_filtered()

    presets = ["UPI", "OTP", "server", "fraud", "money deducted", "login",
               "refund", "crash"]
    c1, c2 = st.columns([2, 2])
    term = c1.text_input("Issue term", value="UPI",
                         help="Substring match on the model text (case-insensitive)")
    preset = c2.selectbox("Or pick a preset", [""] + presets)
    if preset:
        term = preset

    if not term:
        st.info("Enter a term to explore.")
        return

    hit = df["review_text_model"].str.contains(term, case=False, na=False,
                                               regex=False)
    view = df[hit]
    st.caption(f"{len(view):,} of {len(df):,} filtered reviews mention "
               f"“{term}” ({len(view) / max(len(df), 1) * 100:.2f}%)")

    if view.empty:
        st.warning("No mentions in the current filter. Widen the sidebar "
                   "filters or try another term.")
        return

    m = st.columns(4)
    m[0].metric("Mentions", f"{len(view):,}")
    m[1].metric("Banks affected", view["bank"].nunique())
    m[2].metric("Negative rate",
                f"{(view['sentiment'] == 'NEGATIVE').mean() * 100:.1f}%")
    m[3].metric("Failure rate",
                f"{(view['failure_flag'] == 'YES').mean() * 100:.1f}%")

    a, b = st.columns(2)
    with a:
        st.subheader("Government vs Private")
        bar_chart(view["bank_type"].value_counts(), y_title="Mentions")
    with b:
        st.subheader("Sentiment")
        sentiment_bar_chart(view["sentiment"].value_counts())

    c, d = st.columns(2)
    with c:
        st.subheader("Risk")
        risk = view["primary_risk"].value_counts()
        bar_chart(risk if not risk.empty else
                  pd.Series({"no failure signal": len(view)}),
                  y_title="Mentions")
        st.caption("Risk labels exist only for failure=YES rows.")
    with d:
        st.subheader("Emotion")
        bar_chart(view["emotion"].value_counts(), y_title="Mentions")

    st.subheader("Trend")
    dated = view.dropna(subset=["review_date"]).copy()
    if dated.empty:
        st.info("No dated mentions in this slice.")
    else:
        dated["month"] = (pd.to_datetime(dated["review_date"])
                          .dt.to_period("M").dt.to_timestamp())
        monthly = dated.groupby("month").size()
        line_chart(monthly, y_title="Mentions")
        st.caption("Mention volume per month; compare with the corpus-wide "
                   "volume trend before reading any spike (review volume "
                   "itself changes over time).")

    st.subheader("Language distribution")
    st.dataframe(
        view.groupby("language")["sentiment"].value_counts()
        .unstack(fill_value=0).fillna(0).astype(int),
        width="stretch")

    st.subheader("Aspects mentioned alongside")
    aspects = (view["aspects"].dropna().str.split(",").explode()
               .value_counts().head(15))
    if not aspects.empty:
        bar_chart(aspects, horizontal=True, y_title="Mentions")

    st.subheader("Representative reviews")
    mode = st.radio("Selection", ["Most severe", "Most negative", "Most recent"],
                    horizontal=True)
    if mode == "Most severe":
        rep = view.sort_values("severity", ascending=False)
    elif mode == "Most negative":
        rep = view.sort_values(["sentiment_score", "severity"],
                               ascending=[True, False])
    else:
        rep = view.sort_values("review_date", ascending=False,
                               na_position="last")
    review_list(rep, limit=10, caption=f"{mode} mentions of “{term}”")


if __name__ == "__main__":
    render()
else:
    render()
