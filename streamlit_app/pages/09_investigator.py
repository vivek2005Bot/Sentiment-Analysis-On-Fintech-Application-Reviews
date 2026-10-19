"""Page: Review Investigator (plan s.49)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import bar_chart, get_filtered, pii, review_card, \
    sentiment_bar_chart


def render() -> None:
    st.title("🔍 Review Investigator")
    st.caption("Search and filter individual reviews; every displayed review "
               "is PII-masked (plan Rule 11).")
    df = get_filtered()

    c1, c2 = st.columns([2, 1])
    query = c1.text_input("Keyword / phrase (matched on model text)",
                          placeholder="upi, otp, server, fraud, money deducted")
    mode = c2.radio("Match mode", ["contains", "exact word"], horizontal=True)

    c3, c4, c5 = st.columns(3)
    f_fail = c3.selectbox("Failure", ["Any", "YES", "NO"])
    f_risk = c4.selectbox("Primary risk",
                          ["Any", "TECHNICAL_PERFORMANCE_RISK",
                           "FINANCIAL_SECURITY_RISK", "(null / non-failure)"])
    f_sev = c5.selectbox("Min severity", ["Any", 1, 2, 3, 4, 5])

    view = df
    if query:
        if mode == "exact word":
            view = view[view["review_text_model"].str.contains(
                rf"\b{query}\b", case=False, na=False, regex=True)]
        else:
            view = view[view["review_text_model"].str.contains(
                query, case=False, na=False, regex=False)]
    if f_fail != "Any":
        view = view[view["failure_flag"] == f_fail]
    if f_risk == "(null / non-failure)":
        view = view[view["primary_risk"].isna()]
    elif f_risk != "Any":
        view = view[view["primary_risk"] == f_risk]
    if f_sev != "Any":
        view = view[view["severity"] >= int(f_sev)]

    st.caption(f"{len(view):,} matching reviews")

    if query and not view.empty:
        st.subheader("Issue snapshot for this keyword")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mentions", f"{len(view):,}")
        m2.metric("Banks affected", view["bank"].nunique())
        m3.metric("Negative rate",
                  f"{(view['sentiment'] == 'NEGATIVE').mean() * 100:.1f}%")
        m4.metric("Failure rate",
                  f"{(view['failure_flag'] == 'YES').mean() * 100:.1f}%")
        c6, c7 = st.columns(2)
        with c6:
            bar_chart(view["bank_type"].value_counts(), y_title="Reviews")
        with c7:
            sentiment_bar_chart(view["sentiment"].value_counts())

    sort_by = st.selectbox("Sort by", ["severity", "review_date", "bank"])
    asc = st.checkbox("Ascending", value=False)
    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=asc, na_position="last")

    rows = view.head(500)
    st.download_button(
        "Download these results (CSV, PII-masked)",
        data=rows.assign(review_text=pii(rows["review_text"]),
                         reply_text=pii(rows["reply_text"])
                         if "reply_text" in rows else None)
        .to_csv(index=False).encode("utf-8"),
        file_name="investigator_results.csv",
        mime="text/csv",
    )

    for _, row in view.head(25).iterrows():
        review_card(row)


if __name__ == "__main__":
    render()
else:
    render()
