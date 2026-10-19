"""Page: Government vs Private (plan s.49, s.55 Table 5)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import figure, load_table, missing_table_notice, section_header


def render() -> None:
    st.markdown("""
    <h1 style="font-weight: 800; letter-spacing: -0.02em;">
        ⚖️ Government vs Private
    </h1>
    <p style="opacity: 0.55; font-size: 0.85rem; margin-top: -8px;">
        All comparisons are OBSERVATIONAL. Wording is
        'associated with' / 'observed difference'; no causal claims
        (plan Rule 10).
    </p>
    """, unsafe_allow_html=True)

    t = load_table("govt_vs_private.csv")
    if t.empty:
        missing_table_notice("govt_vs_private.csv")
    else:
        section_header("📊", "Statistical Comparison")
        st.dataframe(t, use_container_width=True, hide_index=True)
        st.caption("p-values corrected with Benjamini-Hochberg "
                   "(q_value_bh). Effect size = Cramer's V (categorical) or "
                   "rank-biserial (rating).")

    st.divider()
    section_header("🏦", "Bank Summary")
    bs = load_table("bank_summary.csv")
    if not bs.empty:
        st.dataframe(bs, use_container_width=True, hide_index=True)
    else:
        missing_table_notice("bank_summary.csv")

    st.divider()
    section_header("📈", "Industry-level Improvement Matrix (s.51)")
    bim = load_table("bank_improvement_matrix.csv")
    if bim.empty:
        missing_table_notice("bank_improvement_matrix.csv")
    else:
        banks = sorted(bim["bank"].unique())
        pick = st.multiselect("Filter bank", banks, default=banks[:3])
        view = bim[bim["bank"].isin(pick)] if pick else bim
        view = view.reindex(view["difference"].abs().sort_values(
            ascending=False).index)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("bank_rate vs industry_rate (rest of corpus) over OBSERVED "
                   "issue categories only; suggested_operational_area maps "
                   "the observed category, it is not internal bank knowledge.")

    st.divider()
    section_header("🖼️", "Precomputed Comparison Figures")
    fc1, fc2 = st.columns(2)
    with fc1:
        figure("03_govt_vs_private_sentiment.png")
    with fc2:
        figure("13_pagerank_govt_vs_private.png")


if __name__ == "__main__":
    render()
else:
    render()
