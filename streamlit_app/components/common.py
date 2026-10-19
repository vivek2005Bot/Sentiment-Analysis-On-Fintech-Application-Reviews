"""Shared dashboard components: cached loaders, global filters, PII masking.

Plan constraints honoured here:
- Rule 12: pages read PRECOMPUTED artifacts only (no training/aggregation).
- Rule 11: every review shown passes through mask_pii().
- s.58: KPI helpers state their denominator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import mask_pii  # noqa: E402

PREDICTIONS = ROOT / "data" / "predictions" / "master_predictions.parquet"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Premium CSS injection (called once from app.py)
# ---------------------------------------------------------------------------
def inject_premium_css() -> None:
    """Inject custom CSS for a premium, dark-themed dashboard look."""
    st.markdown("""
    <style>
    /* ---- Import premium font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global typography ---- */
    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Streamlit metric cards ---- */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 18px 22px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.06);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.12);
        border-color: rgba(99, 102, 241, 0.3);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.75;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 1.85rem !important;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ---- Sidebar polish ---- */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(99, 102, 241, 0.1);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1 {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ---- Expander cards ---- */
    [data-testid="stExpander"] {
        border: 1px solid rgba(99, 102, 241, 0.12) !important;
        border-radius: 14px !important;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.08);
    }

    /* ---- DataFrames ---- */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }

    /* ---- Dividers ---- */
    [data-testid="stHorizontalBlock"] + hr {
        border-color: rgba(99, 102, 241, 0.12) !important;
    }

    /* ---- Page title style ---- */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* ---- Subheader style ---- */
    h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }

    /* ---- Button styling ---- */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
    }

    /* ---- Tab styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
    }

    /* ---- Download button ---- */
    .stDownloadButton > button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        color: #10b981 !important;
    }

    /* ---- Selectbox / multiselect ---- */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        border-radius: 10px !important;
    }

    /* ---- Warning / Info / Success / Error boxes ---- */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* ---- Smooth scrollbar ---- */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.2);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Loading precomputed predictions...")
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        st.error("`data/predictions/master_predictions.parquet` not found. "
                 "Run `python -m src.pipeline` first.")
        st.stop()
    return pd.read_parquet(PREDICTIONS)


@st.cache_data(ttl=600)
def load_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def get_filtered() -> pd.DataFrame:
    """Return the globally filtered frame (set by app.py each run)."""
    return st.session_state.get("filtered_df", load_predictions())


def pii(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").map(mask_pii)


# ---------------------------------------------------------------------------
# KPI cards (premium version)
# ---------------------------------------------------------------------------
def kpi_cards(df: pd.DataFrame) -> None:
    """Core KPIs; every help string states its denominator (plan s.58)."""
    n = max(len(df), 1)
    row1 = st.columns(4)
    row1[0].metric("Total Reviews", f"{len(df):,}",
                   help="Rows after sidebar filters / rows in master")
    row1[1].metric("Negative Rate",
                   f"{(df['sentiment'] == 'NEGATIVE').mean() * 100:.1f}%",
                   help="NEGATIVE sentiment / all filtered reviews")
    row1[2].metric("Failure Rate",
                   f"{(df['failure_flag'] == 'YES').mean() * 100:.1f}%",
                   help="failure_flag=YES / all filtered reviews")
    row1[3].metric("Financial Risk Rate",
                   f"{(df['primary_risk'] == 'FINANCIAL_SECURITY_RISK').mean() * 100:.2f}%",
                   help="FINANCIAL_SECURITY_RISK / all filtered reviews "
                        "(Tier-2 label is null when failure=NO)")
    row2 = st.columns(4)
    row2[0].metric("Technical Risk Rate",
                   f"{(df['primary_risk'] == 'TECHNICAL_PERFORMANCE_RISK').mean() * 100:.2f}%",
                   help="TECHNICAL_PERFORMANCE_RISK / all filtered reviews")
    row2[1].metric("Friction Rate",
                   f"{(df['operational_friction'] == 'YES').mean() * 100:.1f}%",
                   help="operational_friction=YES / all filtered reviews")
    row2[2].metric("Critical Failure Rate",
                   f"{df['critical_failure'].mean() * 100:.2f}%",
                   help="critical_failure=True / all filtered reviews")
    row2[3].metric("Banks", f"{df['bank'].nunique()}",
                   help=f"Distinct banks in the filtered slice of {n:,} rows")


# ---------------------------------------------------------------------------
# Review card
# ---------------------------------------------------------------------------
def review_card(row: pd.Series) -> None:
    """Expandable, PII-masked single review with all engine outputs."""
    sentiment = row.get('sentiment', '?')
    sent_emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}.get(
        sentiment, "⚪")
    label = (f"{sent_emoji} {row.get('bank', '?')} · ⭐{row.get('rating', '?')} · "
             f"{sentiment} · "
             f"{str(row.get('review_date', ''))[:10]}")
    with st.expander(label):
        st.write(pii(pd.Series([row.get("review_text", "")]))[0])
        fields = {
            "Bank type": row.get("bank_type"),
            "Language": row.get("language"),
            "Script": row.get("script"),
            "Code-mixed": row.get("is_code_mixed"),
            "Transliterated": row.get("is_transliterated"),
            "Failure": row.get("failure_flag"),
            "Failure confidence": row.get("failure_confidence"),
            "Primary risk": row.get("primary_risk"),
            "Secondary risk": row.get("secondary_risk"),
            "Risk overlap": row.get("risk_overlap"),
            "Critical failure": row.get("critical_failure"),
            "Emotion": row.get("emotion"),
            "Severity": row.get("severity"),
            "Aspects": row.get("aspects"),
            "Aspect sentiment": row.get("aspect_sentiment"),
            "Response bucket": row.get("response_bucket"),
            "Response time (h)": row.get("response_time_hours"),
            "Model source": row.get("sentiment_model_source"),
        }
        cols = st.columns(3)
        for i, (k, v) in enumerate(fields.items()):
            shown = v if pd.notna(v) else "—"
            cols[i % 3].markdown(f"**{k}:** {shown}")
        reply = row.get("reply_text")
        if pd.notna(reply) and str(reply).strip() not in ("", "nan", "None"):
            st.markdown("**Developer reply:**")
            st.info(pii(pd.Series([reply]))[0])


def review_list(df: pd.DataFrame, limit: int = 25, caption: str = "") -> None:
    if caption:
        st.caption(f"{caption} · {len(df):,} matching · showing {min(limit, len(df))} · PII-masked")
    for _, row in df.head(limit).iterrows():
        review_card(row)


def figure(name: str, caption: str = "") -> None:
    """Show a precomputed figure from reports/figures (never recomputed)."""
    path = FIGURES / name
    if not path.exists():
        st.info(f"Figure `{name}` not generated yet - run "
                "`python -m src.figures`.")
        return
    st.image(str(path), use_container_width=True)
    if caption:
        st.caption(caption)


def missing_table_notice(name: str) -> st.DeltaGenerator:
    st.warning(f"`reports/tables/{name}` not found - run the full pipeline "
               "(`python -m src.pipeline`).")


# ---------------------------------------------------------------------------
# Shared Plotly chart helpers (prettier than bare st.bar_chart/st.line_chart).
# All helpers are display-only: they read the ALREADY-FILTERED frame and never
# recompute engine outputs (plan Rule 12).
# ---------------------------------------------------------------------------
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# Premium color palette
PALETTE = [
    "#818cf8",  # Indigo 400
    "#f472b6",  # Pink 400
    "#34d399",  # Emerald 400
    "#fbbf24",  # Amber 400
    "#60a5fa",  # Blue 400
    "#a78bfa",  # Violet 400
    "#fb923c",  # Orange 400
    "#2dd4bf",  # Teal 400
    "#e879f9",  # Fuchsia 400
    "#f87171",  # Red 400
]

GRADIENT_COLORS = {
    "indigo": ["#6366f1", "#818cf8", "#a5b4fc"],
    "emerald": ["#059669", "#34d399", "#6ee7b7"],
    "rose": ["#e11d48", "#fb7185", "#fda4af"],
    "amber": ["#d97706", "#fbbf24", "#fde68a"],
    "violet": ["#7c3aed", "#a78bfa", "#c4b5fd"],
}


def _style(fig: "go.Figure", y_title: str = "") -> "go.Figure":
    """Consistent, premium look across every dashboard chart."""
    fig.update_layout(
        margin=dict(l=24, r=24, t=56, b=24),
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06,
            xanchor="center", x=0.5,
            font=dict(size=12, color="rgba(156, 163, 175, 0.85)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(size=13, family="Inter, -apple-system, sans-serif",
                  color="rgba(156, 163, 175, 0.85)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(30, 30, 46, 0.95)",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="#e2e8f0",
            bordercolor="rgba(99, 102, 241, 0.3)",
        ),
        title_font=dict(size=17, color="#e2e8f0"),
    )
    if y_title:
        fig.update_yaxes(
            title_text=y_title,
            title_font=dict(size=12, color="rgba(156, 163, 175, 0.6)"),
            gridcolor="rgba(99, 102, 241, 0.06)",
            zerolinecolor="rgba(99, 102, 241, 0.12)",
            tickfont=dict(color="rgba(156, 163, 175, 0.6)"),
        )
    else:
        fig.update_yaxes(
            title_text=None,
            gridcolor="rgba(99, 102, 241, 0.06)",
            zerolinecolor="rgba(99, 102, 241, 0.12)",
            tickfont=dict(color="rgba(156, 163, 175, 0.6)"),
        )

    fig.update_xaxes(
        title_text=None,
        gridcolor="rgba(99, 102, 241, 0.06)",
        zerolinecolor="rgba(99, 102, 241, 0.12)",
        tickfont=dict(color="rgba(156, 163, 175, 0.6)"),
    )
    return fig


def bar_chart(series: pd.Series, title: str = "", y_title: str = "Count",
              color: str | None = None, horizontal: bool = False) -> None:
    """Styled bar chart from a value-counts-like Series (index = category)."""
    if series is None or (hasattr(series, "empty") and series.empty):
        st.info("No data in the current filter.")
        return
    s = series.dropna()
    long_df = s.rename_axis("category").reset_index(name="value")
    long_df["category"] = long_df["category"].astype(str)
    if horizontal:
        fig = px.bar(long_df, x="value", y="category", orientation="h",
                     color_discrete_sequence=[color or PALETTE[0]])
        fig.update_yaxes(autorange="reversed")
    else:
        fig = px.bar(long_df, x="category", y="value",
                     color_discrete_sequence=[color or PALETTE[0]])
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=6,
        opacity=0.9,
        hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>" if horizontal else "<b>%{x}</b><br>%{y:,}<extra></extra>")
    fig = _style(fig, y_title)
    if title:
        fig.update_layout(title_text=f"<b>{title}</b>", title_x=0.01,
                          title_font=dict(size=17))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def grouped_bar_chart(df: pd.DataFrame, title: str = "",
                      y_title: str = "Count") -> None:
    """Styled grouped bar chart: index becomes x-axis, one trace per column."""
    if df is None or df.empty:
        st.info("No data in the current filter.")
        return
    long_df = df.rename_axis("category").reset_index()
    long_df["category"] = long_df["category"].astype(str)
    fig = px.bar(long_df, x="category",
                 y=list(df.columns), barmode="group",
                 color_discrete_sequence=PALETTE)
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=5,
        opacity=0.9,
    )
    fig = _style(fig, y_title)
    if title:
        fig.update_layout(title_text=f"<b>{title}</b>", title_x=0.01,
                          title_font=dict(size=17))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def line_chart(df: pd.DataFrame | pd.Series, title: str = "",
               y_title: str = "Count") -> None:
    """Styled multi-series line chart from a DataFrame (index = x-axis)."""
    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("No data in the current filter.")
        return
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name or "value")
    long_df = df.reset_index()
    x_col = df.index.name or "index"
    long_df[x_col] = long_df[x_col].astype(str)
    fig = px.line(long_df, x=x_col,
                  y=list(df.columns), markers=True,
                  color_discrete_sequence=PALETTE)
    fig.update_traces(
        line=dict(width=3),
        mode="lines+markers",
        marker=dict(size=7, line=dict(width=2, color="rgba(0,0,0,0.3)")),
        line_shape="spline",
        fill="tozeroy",
        fillcolor="rgba(129, 140, 248, 0.08)",
    )
    fig = _style(fig, y_title)
    if title:
        fig.update_layout(title_text=f"<b>{title}</b>", title_x=0.01,
                          title_font=dict(size=17))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def sentiment_bar_chart(series: pd.Series, title: str = "",
                        y_title: str = "Count") -> None:
    """Bar chart with the project's sentiment colour coding."""
    colors = {"POSITIVE": "#34d399", "NEUTRAL": "#94a3b8",
              "NEGATIVE": "#f87171"}
    s = series.dropna()
    long_df = s.rename_axis("category").reset_index(name="value")
    long_df["category"] = long_df["category"].astype(str)
    fig = px.bar(long_df, x="category", y="value",
                 color="category",
                 color_discrete_map=colors | {c: PALETTE[0]
                                              for c in long_df["category"]
                                              if c not in colors})
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=6,
        opacity=0.9,
        hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>")
    fig = _style(fig, y_title)
    if title:
        fig.update_layout(title_text=f"<b>{title}</b>", title_x=0.01,
                          title_font=dict(size=17))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def donut_chart(series: pd.Series, title: str = "",
                hole: float = 0.55) -> None:
    """Premium donut chart from a value-counts-like Series."""
    if series is None or series.empty:
        st.info("No data in the current filter.")
        return
    s = series.dropna()
    fig = go.Figure(data=[go.Pie(
        labels=s.index.astype(str),
        values=s.values,
        hole=hole,
        textinfo="label+percent",
        textposition="outside",
        marker=dict(
            colors=PALETTE[:len(s)],
            line=dict(color="rgba(0,0,0,0.15)", width=1),
        ),
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        margin=dict(l=24, r=24, t=56, b=24),
        height=380,
        font=dict(size=13, family="Inter, sans-serif",
                  color="rgba(156, 163, 175, 0.85)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=11, color="rgba(156, 163, 175, 0.85)"),
        ),
        hoverlabel=dict(
            bgcolor="rgba(30, 30, 46, 0.95)",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="#e2e8f0",
            bordercolor="rgba(99, 102, 241, 0.3)",
        ),
    )
    if title:
        fig.update_layout(title_text=f"<b>{title}</b>", title_x=0.01,
                          title_font=dict(size=17, color="#e2e8f0"))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a styled section header with optional subtitle."""
    st.markdown(f"### {icon} {title}")
    if subtitle:
        st.caption(subtitle)
