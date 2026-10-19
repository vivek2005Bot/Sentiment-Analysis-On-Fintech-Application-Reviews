"""Page: Critical Review Queue (plan s.49) - the human-action worklist."""
from __future__ import annotations

import streamlit as st

from components.common import bar_chart, get_filtered, load_table, review_list


def render() -> None:
    st.title("🚨 Critical Review Queue")
    st.caption("The prioritised human-action worklist: critical failures and/or "
               "financial-security risk, with confidence, severity and "
               "response status (plan s.49).")
    df = get_filtered()

    fin = df["primary_risk"] == "FINANCIAL_SECURITY_RISK"
    crit = df["critical_failure"] == True  # noqa: E712
    queue = df[fin | crit].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queue size", f"{len(queue):,}",
              help="(financial risk OR critical failure) / all filtered reviews")
    c2.metric("Financial security", f"{int(fin.sum()):,}")
    c3.metric("Critical failure", f"{int(crit.sum()):,}")
    unanswered = queue[queue["has_developer_response"] != True]  # noqa: E712
    c4.metric("Unanswered", f"{len(unanswered):,}",
              help="Queue rows with no developer reply")

    st.subheader("Queue composition")
    a, b = st.columns(2)
    with a:
        st.write("By bank")
        bar_chart(queue["bank"].value_counts(), horizontal=True)
    with b:
        st.write("By severity")
        bar_chart(queue["severity"].value_counts().sort_index(),
                  y_title="Queue items")

    st.subheader("Automated triage routing (plan s.39)")
    conf_threshold = st.slider(
        "Low-confidence threshold for human review", 0.0, 1.0, 0.70, 0.05,
        help="Config default human_review.confidence_threshold=0.70")
    low_conf = queue[
        (queue["primary_risk_confidence"].fillna(1.0) < conf_threshold)
        | (queue["risk_overlap"] == "YES")]
    st.caption(f"{len(low_conf):,} queue rows are low-confidence or "
               "mixed-risk and should be human-reviewed first.")
    if not low_conf.empty:
        st.dataframe(
            low_conf[["bank", "primary_risk", "secondary_risk",
                      "primary_risk_confidence", "severity",
                      "response_bucket", "review_text"]].head(50),
            width="stretch")

    st.subheader("Full queue preview (sorted by severity, then confidence)")
    sorted_q = queue.sort_values(
        ["severity", "primary_risk_confidence"],
        ascending=[False, False], na_position="last")
    review_list(sorted_q, limit=20, caption="Queue preview")

    st.download_button(
        "Export full queue metadata (CSV)",
        data=sorted_q[["review_id", "bank", "bank_type", "rating",
                       "review_date", "sentiment", "failure_flag",
                       "primary_risk", "secondary_risk", "risk_overlap",
                       "severity", "primary_risk_confidence",
                       "response_bucket"]].to_csv(index=False).encode("utf-8"),
        file_name="critical_review_queue.csv",
        mime="text/csv",
    )
    st.caption("Review text is intentionally excluded from this export - use "
               "the Investigator page for PII-masked text.")


if __name__ == "__main__":
    render()
else:
    render()
