"""Indian Banking Review Intelligence - Streamlit decision-support dashboard.

Plan compliance:
- Rule 12: pages read PRECOMPUTED artifacts only; no training or heavy
  aggregation happens on page load.
- Rule 11: all review text passes through mask_pii() before display.
- s.57: global sidebar filters; each page recomputes denominators after
  filtering (KPI helpers state their denominators - s.58).
- s.48: pages live under streamlit_app/pages/ and are wired explicitly via
  st.navigation, so nothing is auto-discovered twice.

Run: .venv/bin/streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (str(ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

st.set_page_config(
    page_title="Indian Banking Review Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _build_sidebar_filters(df):
    """Global filters (plan s.57). Returns the filtered frame."""
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 8px 0 4px 0;">
        <span style="font-size: 2.2rem;">🏦</span>
        <h2 style="margin: 4px 0 0 0; font-weight: 800; letter-spacing: -0.02em;
                   background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   background-clip: text; font-size: 1.3rem;">
            Review Intelligence
        </h2>
        <p style="margin: 2px 0 0 0; font-size: 0.72rem; opacity: 0.5; letter-spacing: 0.05em;">
            MULTILINGUAL BANKING ANALYTICS
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.markdown("##### 🔍 Global Filters")

    f = df
    banks = st.sidebar.multiselect(
        "Bank", sorted(df["bank"].dropna().unique().tolist()))
    if banks:
        f = f[f["bank"].isin(banks)]

    types = st.sidebar.multiselect(
        "Bank type", sorted(df["bank_type"].dropna().unique().tolist()))
    if types:
        f = f[f["bank_type"].isin(types)]

    langs = st.sidebar.multiselect(
        "Language", sorted(df["language"].dropna().unique().tolist()))
    if langs:
        f = f[f["language"].isin(langs)]

    sentiments = st.sidebar.multiselect(
        "Sentiment", ["POSITIVE", "NEUTRAL", "NEGATIVE"])
    if sentiments:
        f = f[f["sentiment"].isin(sentiments)]

    risks = st.sidebar.multiselect(
        "Primary risk", ["TECHNICAL_PERFORMANCE_RISK",
                         "FINANCIAL_SECURITY_RISK"])
    if risks:
        f = f[f["primary_risk"].isin(risks)]

    severities = st.sidebar.multiselect("Severity", [1, 2, 3, 4, 5])
    if severities:
        f = f[f["severity"].isin(severities)]

    ratings = st.sidebar.multiselect("Rating", [1, 2, 3, 4, 5])
    if ratings:
        f = f[f["rating"].isin(ratings)]

    if st.sidebar.button("🔄 Reset filters", use_container_width=True):
        st.session_state.pop("filtered_df", None)
        st.rerun()

    st.sidebar.divider()

    # Stats badge
    pct_selected = len(f) / max(len(df), 1) * 100
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(168,85,247,0.1) 100%);
                border: 1px solid rgba(99,102,241,0.15); border-radius: 12px; padding: 14px; text-align: center;">
        <div style="font-size: 1.6rem; font-weight: 700;
                    background: linear-gradient(135deg, #818cf8, #a78bfa);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
            {len(f):,}
        </div>
        <div style="font-size: 0.7rem; opacity: 0.5; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px;">
            of {len(df):,} reviews ({pct_selected:.1f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.caption(
        "All displayed review text is PII-masked. Metrics are observational; "
        "no causal claims are made."
    )
    return f


def main() -> None:
    from streamlit_app.components.common import inject_premium_css, load_predictions

    inject_premium_css()
    df_all = load_predictions()
    st.session_state["filtered_df"] = _build_sidebar_filters(df_all)

    pages = [
        st.Page("pages/01_overview.py", title="Executive Overview",
                icon="🏦", default=True),
        st.Page("pages/02_sentiment.py", title="Sentiment Intelligence",
                icon="🙂"),
        st.Page("pages/03_risk.py", title="Risk Intelligence", icon="⚠️"),
        st.Page("pages/04_govt_private.py", title="Government vs Private",
                icon="⚖️"),
        st.Page("pages/05_language.py", title="Language & Hinglish",
                icon="🌐"),
        st.Page("pages/06_benchmark.py", title="Model Benchmark", icon="🤖"),
        st.Page("pages/07_aspects.py", title="Aspect & PageRank", icon="🧩"),
        st.Page("pages/08_response.py", title="Response Intelligence",
                icon="💬"),
        st.Page("pages/13_issue_explorer.py", title="Issue Explorer",
                icon="🧭"),
        st.Page("pages/09_investigator.py", title="Review Investigator",
                icon="🔍"),
        st.Page("pages/10_critical_queue.py", title="Critical Review Queue",
                icon="🚨"),
        st.Page("pages/11_explainability.py", title="Model Explainability",
                icon="🔎"),
        st.Page("pages/12_early_warning.py", title="Early Warning", icon="📈"),
        st.Page("pages/14_methodology.py", title="Data & Methodology",
                icon="📚"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
