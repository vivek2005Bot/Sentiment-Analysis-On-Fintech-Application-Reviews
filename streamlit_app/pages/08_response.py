"""Page: Response Intelligence (plan s.49, s.35)."""
from __future__ import annotations

import streamlit as st

from components.common import bar_chart, figure, get_filtered, load_table, \
    review_list


def render() -> None:
    st.title("💬 Response Intelligence")
    st.caption("Enabled only because reply metadata exists in this corpus. "
               "All latency findings are OBSERVATIONAL associations, never "
               "causal recovery claims (plan s.35 / Rule 10).")
    df = get_filtered()

    responded = df[df["has_developer_response"] == True]  # noqa: E712
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews with a reply", f"{len(responded):,}",
              help="Developer reply present / all filtered reviews")
    med = df["response_time_hours"].median()
    c2.metric("Median latency", f"{med:.1f} h" if med == med else "—",
              help="Median of parsed reply-minus-review hours")
    c3.metric("Within 24h",
              f"{(df['response_bucket'] == '<24h').mean() * 100:.1f}%",
              help="<24h bucket / all filtered reviews")
    c4.metric("No response",
              f"{(df['response_bucket'] == 'No response').mean() * 100:.1f}%",
              help="No reply_text / all filtered reviews")

    st.subheader("Response bucket table (s.55 Table 7)")
    t = load_table("response_analysis.csv")
    if t.empty:
        st.warning("response_analysis.csv not found - run the full pipeline.")
    else:
        st.dataframe(t, width="stretch")
        st.caption("Denominator: reviews inside each response bucket. "
                   "kpi notes: mean_rating and negative_rate are levels, not "
                   "changes - this corpus has no follow-up review snapshots, "
                   "so rating/sentiment CHANGE cannot be computed.")

    st.subheader("Latency distribution")
    h = df["response_time_hours"].dropna()
    if not h.empty:
        bar_chart(h.clip(upper=200).round(0).value_counts().sort_index(),
                  y_title="Reviews")

    st.subheader("Response status by risk class")
    if df["primary_risk"].notna().any():
        piv = (df[df["primary_risk"].notna()]
               .assign(responded=df["has_developer_response"] == True)  # noqa: E712
               .groupby("primary_risk")["responded"].mean().mul(100))
        bar_chart(piv, y_title="Responded (%)")
        st.caption("% of that risk class that received any developer reply.")

    st.divider()
    figure("14_response_latency_distribution.png")
    figure("15_response_latency_vs_rating.png")

    unanswered_risk = df[(df["primary_risk"] == "FINANCIAL_SECURITY_RISK")
                         & (df["has_developer_response"] != True)]  # noqa: E712
    st.subheader("Financial-risk reviews without a developer reply")
    review_list(unanswered_risk, limit=10,
                caption="Highest-priority operational follow-up")


if __name__ == "__main__":
    render()
else:
    render()
