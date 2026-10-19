"""Page: Model Benchmark (plan s.49, s.55 Table 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _ROOT / "streamlit_app"
for _p in (str(_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from components.common import figure, load_table, section_header
except ModuleNotFoundError:  # standalone `streamlit run` of this file
    from streamlit_app.components.common import figure, load_table, section_header


def render() -> None:
    st.markdown("""
    <h1 style="font-weight: 800; letter-spacing: -0.02em;">
        🤖 Model Benchmark
    </h1>
    <p style="opacity: 0.55; font-size: 0.85rem; margin-top: -8px;">
        Only held-out metrics are shown (plan Rule 9). Baselines are
        clearly separated from transformer results.
    </p>
    """, unsafe_allow_html=True)

    bench = load_table("model_benchmark.csv")
    if bench.empty:
        st.info("No transformer benchmark yet. Train a task first, e.g.:\n\n"
                "```bash\n.venv/bin/python -m src.train --task sentiment "
                "--model distilbert-base-uncased\n```")
    else:
        section_header("🎯", "Transformer Results (held-out)")
        st.dataframe(bench, use_container_width=True, hide_index=True)
        st.caption("macro_f1 is the selection metric; per-class recall matters "
                   "more than accuracy for imbalanced risk classes (s.22).")
        if "task" in bench.columns:
            tasks = sorted({str(v).strip() for v in
                            bench["task"].dropna().unique()
                            if str(v).strip()})
        else:
            tasks = []
        if tasks:
            pick = st.selectbox("Task detail", ["All"] + tasks)
            view = (bench if pick == "All"
                    else bench[bench["task"].astype(str) == pick])
            st.dataframe(view, use_container_width=True, hide_index=True)

    st.divider()
    section_header("🌐", "Language-specific Model Performance (s.25)")
    lang = load_table("language_model_performance.csv")
    if lang.empty:
        st.info("Not generated yet - produced automatically when a model is "
                "trained (held-out test split).")
    else:
        st.dataframe(lang, use_container_width=True, hide_index=True)
        st.caption("Small subsets are excluded (<20 rows) to avoid unstable "
                   "estimates.")

    st.divider()
    section_header("📐", "Flat vs Hierarchical Architecture (s.24)")
    fvh = load_table("flat_vs_hierarchical.csv")
    if fvh.empty:
        st.info("Run `python -m src.evaluate` after training the flat head to "
                "populate this comparison.")
    else:
        st.dataframe(fvh, use_container_width=True, hide_index=True)
        st.caption("An improvement is claimed only when both approaches are "
                   "measured on the same held-out rows.")

    st.divider()
    section_header("📋", "Baseline (rule-based) Performance")
    st.markdown("""
The deterministic engines are **baselines for bootstrapping**, not the final
classifier (plan Rule 8). Their `model_source` values are recorded per row:

| Component | model_source | Status |
|---|---|---|
| Sentiment | `rules_lexicon_rating_v1` | baseline; replaced by transformer |
| Failure gate | `rules_lexicon_v1` | baseline; silver-label source |
| Tier-2 risk | `priority_financial_first_v1` | baseline; priority rule s.14 |
| Emotion | `lexicon_v1` | baseline |

Labels built from these are marked `label_source=silver_rules_v1` in the
annotation set and are **provisional** until human labels exist.
""")

    st.divider()
    section_header("🖼️", "Precomputed Benchmark Figures")
    fc1, fc2 = st.columns(2)
    with fc1:
        figure("16_model_benchmark.png")
        figure("18_language_specific_model_performance.png")
    with fc2:
        figure("17_confusion_matrices.png")
        figure("19_flat_vs_hierarchical.png")


render()
