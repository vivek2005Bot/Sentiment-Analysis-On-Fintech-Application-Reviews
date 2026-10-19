"""Page: Language & Hinglish (plan s.49, s.25)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _ROOT / "streamlit_app"
for _p in (str(_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from streamlit_app.components.common import (bar_chart, donut_chart, figure,
                                             get_filtered, load_table,
                                             missing_table_notice,
                                             review_list, section_header)


def render() -> None:
    st.markdown("""
    <h1 style="font-weight: 800; letter-spacing: -0.02em;">
        🌐 Language & Hinglish
    </h1>
    <p style="opacity: 0.55; font-size: 0.85rem; margin-top: -8px;">
        Latin script NEVER implies English (plan s.9): romanized
        Hinglish/Tanglish is detected with a transliteration lexicon;
        native-script reviews are classified by Unicode script.
    </p>
    """, unsafe_allow_html=True)
    df = get_filtered()

    c1, c2, c3 = st.columns(3)
    c1.metric("🌍 Languages Detected", df["language"].nunique())
    c2.metric("🔄 Code-mixed Reviews",
              f"{(df['is_code_mixed'] == True).sum():,}",  # noqa: E712
              help="Mixed-script or romanized+English / all filtered reviews")
    c3.metric("📝 Transliterated",
              f"{(df['is_transliterated'] == True).sum():,}",  # noqa: E712
              help="Romanized Indic without native script")

    st.divider()
    l1, l2 = st.columns(2)
    with l1:
        section_header("📊", "Language Distribution")
        lang = load_table("language_summary.csv")
        if lang.empty:
            missing_table_notice("language_summary.csv")
        else:
            st.dataframe(lang, use_container_width=True, hide_index=True)
            st.caption("negative_rate and failure_rate use that language's own "
                       "review count as the denominator.")
    with l2:
        section_header("🔤", "Script Distribution")
        donut_chart(df["script"].value_counts(), title="Scripts")

    st.divider()
    section_header("🎯", "Language × Sentiment")
    st.dataframe(
        df.groupby("language")["sentiment"].value_counts()
        .unstack(fill_value=0).fillna(0).astype(int),
        use_container_width=True)

    section_header("🏦", "Language Coverage by Bank")
    cov = (df.groupby(["bank", "language"]).size().unstack(fill_value=0))
    st.dataframe(cov, use_container_width=True)

    section_header("📏", "Confidence Distribution (heuristic detector)")
    bar_chart(pd.cut(df["language_confidence"].dropna(),
                     bins=[0, 0.5, 0.7, 0.85, 1.0]).value_counts().sort_index(),
              y_title="Reviews")

    st.divider()
    figure("06_risk_by_language.png")

    with st.expander("📖 Language Detection Methodology"):
        st.markdown("""
- `script`: Unicode block with the largest share of letters
  (Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi,
  Gujarati, Oriya, or Latin).
- `language`: native-script majority -> that language; Latin-dominant ->
  romanized-Hindi/Tamil lexicon ratios decide Hinglish/Tanglish/Other_Indian,
  otherwise English.
- `is_code_mixed`: both native script and Latin present, or romanized Indic
  tokens alongside English tokens.
- `is_transliterated`: romanized Indic above threshold without native script.
- `language_confidence`: deterministic heuristic margin (NOT a calibrated
  probability). Reviews under 2 tokens are `Unknown` by design.
""")

    st.divider()
    hinglish = df[df["language"].isin(["Hinglish", "Tanglish"])]
    section_header("💬", "Hinglish/Tanglish Examples")
    review_list(hinglish, limit=10, caption="Code-mixed reviews")


if __name__ == "__main__":
    render()
else:
    render()
