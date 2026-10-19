"""Required visualizations (plan s.56) written to reports/figures/.

All figures are generated offline by src.pipeline; the dashboard only reads
precomputed artifacts (plan Rule 12). Model-dependent figures (confusion
matrices, language-specific performance, flat-vs-hierarchical) are skipped
with a logged reason when no trained artifacts exist yet.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / offline rendering
import matplotlib.pyplot as plt
import pandas as pd

import src.config as sc
from src.utils import setup_logging

logger = setup_logging("figures")

FIGSIZE = (9, 5)
DPI = 140


def _save(fig, name: str) -> Path | None:
    out_dir = sc.get_paths()["figures_dir"]
    path = out_dir / name
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        logger.info("Figure -> %s", name)
        return path
    except Exception as e:  # never let one figure break the pipeline
        logger.warning("Figure %s failed: %s", name, e)
        return None
    finally:
        plt.close(fig)


def _bar(values: pd.Series, title: str, name: str, xlabel="", ylabel="Reviews",
         rot=0, color="#3b6ea5"):
    if values is None or len(values) == 0:
        logger.info("Skipping %s (no data)", name)
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar([str(i) for i in values.index], values.to_numpy(), color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rot)
    for i, v in enumerate(values.to_numpy()):
        ax.annotate(f"{v:,.0f}" if abs(v) >= 1 else f"{v:.3f}",
                    (i, v), ha="center", va="bottom", fontsize=8)
    return _save(fig, name)


def _rate_bar(rate: pd.Series, title: str, name: str, xlabel="", rot=0,
              color="#a5533b"):
    if rate is None or len(rate) == 0:
        logger.info("Skipping %s (no data)", name)
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar([str(i) for i in rate.index], (rate * 100).to_numpy(), color=color)
    ax.set_title(f"{title} (%)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rate (%)")
    ax.tick_params(axis="x", rotation=rot)
    return _save(fig, name)


# --- 1-5 -------------------------------------------------------------------
def fig_sentiment_distribution(master: pd.DataFrame):
    return _bar(master["sentiment"].value_counts(), "Sentiment distribution",
                "01_sentiment_distribution.png", "Sentiment")


def fig_sentiment_over_time(master, freq="M"):
    d = master.dropna(subset=["review_date"]).copy()
    if d.empty:
        logger.info("Skipping 02 (no dates)")
        return None
    d["period"] = pd.to_datetime(d["review_date"]).dt.to_period(freq).dt.to_timestamp()
    g = (d.groupby(["period", "sentiment"]).size().unstack(fill_value=0)
         .sort_index())
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in g.columns:
        ax.plot(g.index, g[col], marker="o", ms=3, label=col)
    ax.set_title("Sentiment over time")
    ax.set_ylabel("Reviews")
    ax.legend()
    return _save(fig, "02_sentiment_over_time.png")


def fig_govt_vs_private_sentiment(master):
    g = (master.groupby("bank_type")["sentiment"]
         .value_counts(normalize=True).unstack(fill_value=0).mul(100))
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    g.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment mix: Government vs Private (%)")
    ax.set_ylabel("Share of reviews (%)")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, "03_govt_vs_private_sentiment.png")


def fig_technical_vs_financial_risk(master):
    return _bar(master["primary_risk"].value_counts(dropna=True),
                "Technical vs Financial risk (failure rows only)",
                "04_technical_vs_financial_risk.png", "Primary risk")


def fig_risk_by_bank(master):
    g = (master.groupby("bank")["primary_risk"]
         .value_counts(normalize=True).unstack(fill_value=0).mul(100))
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    g.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Risk mix by bank (% of that bank's reviews)")
    ax.set_ylabel("Share (%)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "05_risk_by_bank.png")


# --- 6-11, 20 --------------------------------------------------------------
def fig_risk_by_language(master):
    return _bar(master.groupby("language")["primary_risk"].count().sort_values(
        ascending=False).head(12),
        "FAILURE-gate reviews by language (risk is null when failure=NO)",
        "06_risk_by_language.png", "Language", rot=45)


def fig_risk_by_rating(master):
    g = (master.dropna(subset=["rating"]).groupby("rating")["primary_risk"]
         .value_counts(normalize=True).unstack(fill_value=0).mul(100))
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    g.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Risk mix by star rating (%)")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Share (%)")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, "07_risk_by_rating.png")


def fig_emotion_distribution(master):
    return _bar(master["emotion"].value_counts(), "Emotion distribution",
                "08_emotion_distribution.png", "Emotion", rot=45)


def fig_emotion_by_risk(master):
    sub = master[master["primary_risk"].notna()]
    g = (sub.groupby("primary_risk")["emotion"]
         .value_counts(normalize=True).unstack(fill_value=0).mul(100))
    if g.empty:
        logger.info("Skipping 09 (no failure rows)")
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    g.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Emotion mix within each primary risk class (%)")
    ax.set_ylabel("Share (%)")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, "09_emotion_by_risk.png")


def fig_aspect_frequency(master):
    counts = (master["aspects"].dropna().str.split(",").explode()
              .value_counts().head(20))
    return _bar(counts, "Top 20 aspect mentions", "10_aspect_frequency.png",
                "Aspect", rot=60)


def fig_aspect_sentiment(master):
    rows = []
    for aspects, sents in zip(master["aspects"].dropna(),
                              master["aspect_sentiment"].dropna()):
        mapping = dict(p.split(":") for p in str(sents).split(";") if ":" in p)
        for a in str(aspects).split(","):
            if a in mapping:
                rows.append({"aspect": a, "aspect_sentiment": mapping[a]})
    if not rows:
        logger.info("Skipping 11 (no aspect sentiment)")
        return None
    df = pd.DataFrame(rows)
    top = df["aspect"].value_counts().head(12).index
    g = (df[df["aspect"].isin(top)]
         .groupby("aspect")["aspect_sentiment"].value_counts(normalize=True)
         .unstack(fill_value=0).mul(100))
    fig, ax = plt.subplots(figsize=(11, 5))
    g.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Aspect-level sentiment for top aspects (%)")
    ax.set_ylabel("Share (%)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "11_aspect_sentiment.png")


def fig_severity_distribution(master):
    counts = master["severity"].value_counts().sort_index()
    return _bar(counts, "Severity distribution (project policy 1-5)",
                "20_severity_distribution.png", "Severity level", rot=0)


# --- 12-13 -----------------------------------------------------------------
def fig_pagerank_top_aspects():
    path = sc.get_paths()["tables_dir"] / "pagerank_global.csv"
    if not path.exists():
        logger.info("Skipping 12 (no pagerank_global.csv)")
        return None
    df = pd.read_csv(path).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["aspect"][::-1], df["pagerank"][::-1], color="#4b7f52")
    ax.set_title("Top aspects by PageRank (structural centrality)")
    ax.set_xlabel("PageRank")
    return _save(fig, "12_pagerank_top_aspects.png")


def fig_pagerank_govt_vs_private():
    frames = []
    for group in ("GOVERNMENT", "PRIVATE"):
        p = sc.get_paths()["tables_dir"] / f"pagerank_group_{group}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["bank_type"] = group
            frames.append(d)
    if not frames:
        logger.info("Skipping 13 (no group pagerank tables)")
        return None
    df = pd.concat(frames)
    top = df.groupby("aspect")["pagerank"].max().nlargest(10).index
    piv = (df[df["aspect"].isin(top)].pivot_table(
        index="aspect", columns="bank_type", values="pagerank",
        fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 6))
    piv.plot(kind="barh", ax=ax)
    ax.set_title("Aspect centrality: Government vs Private")
    ax.set_xlabel("PageRank")
    return _save(fig, "13_pagerank_govt_vs_private.png")


# --- 14-15 -----------------------------------------------------------------
def fig_response_latency_distribution(master):
    latency = master["response_time_hours"].dropna()
    if latency.empty:
        logger.info("Skipping 14 (no response metadata)")
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(latency.clip(upper=200), bins=60, color="#5a6ea5")
    ax.set_title("Developer response latency (hours, clipped at 200h)")
    ax.set_xlabel("Response time (hours)")
    ax.set_ylabel("Reviews")
    return _save(fig, "14_response_latency_distribution.png")


def fig_response_latency_vs_rating(master):
    """Observational association only - no causal claim (plan Rule 10)."""
    d = master.dropna(subset=["response_time_hours", "rating"])
    if d.empty or d["response_bucket"].dropna().empty:
        logger.info("Skipping 15 (no response metadata)")
        return None
    order = ["<24h", "24-48h", "48-72h", ">72h", "No response"]
    g = (d.groupby("response_bucket")["rating"].mean()
         .reindex([b for b in order if b in set(d["response_bucket"])]))
    rate = (d.assign(neg=d["sentiment"] == "NEGATIVE")
            .groupby("response_bucket")["neg"].mean()
            .reindex(g.index).mul(100))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = range(len(g))
    ax.bar([i - 0.2 for i in x], g.to_numpy(), width=0.4,
           label="Mean rating", color="#4b7f52")
    ax2 = ax.twinx()
    ax2.bar([i + 0.2 for i in x], rate.to_numpy(), width=0.4,
            label="Negative rate %", color="#a5533b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(g.index), rotation=20)
    ax.set_ylabel("Mean rating")
    ax2.set_ylabel("Negative rate (%)")
    ax.set_title("Response bucket vs rating/sentiment (observational)")
    return _save(fig, "15_response_latency_vs_rating.png")


# --- 16-19 (model-dependent; skipped until models are trained) -------------
def fig_model_benchmark():
    path = sc.get_paths()["tables_dir"] / "model_benchmark.csv"
    if not path.exists():
        logger.info("Skipping 16 (no trained-model benchmark yet)")
        return None
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    df["label"] = df["model"].str.split("/").str[-1] + "\n(" + df["task"] + ")"
    ax.bar(df["label"], df["macro_f1"], color="#4b7f52")
    ax.set_title("Model benchmark (held-out macro F1)")
    ax.set_ylabel("Macro F1")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "16_model_benchmark.png")


def fig_confusion_matrices():
    from sklearn.metrics import confusion_matrix

    frames = []
    for task in ("sentiment", "failure", "risk", "emotion", "flat"):
        pred_file = (sc.get_paths()["predictions_dir"] /
                     f"{task}_predictions_eval.parquet")
        if pred_file.exists():
            frames.append(pd.read_parquet(pred_file))
    if not frames:
        logger.info("Skipping 17 (no evaluation prediction dumps yet)")
        return None
    fig, axes = plt.subplots(1, len(frames), figsize=(4.2 * len(frames), 4))
    if len(frames) == 1:
        axes = [axes]
    for ax, df in zip(axes, frames):
        labels = sorted(set(df["y_true"]) | set(df["y_pred"]), key=str)
        cm = confusion_matrix(df["y_true"].astype(int), df["y_pred"].astype(int),
                             labels=list(range(len(labels))))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(str(df["task"].iloc[0]), fontsize=9)
    return _save(fig, "17_confusion_matrices.png")


def fig_language_specific_performance():
    path = sc.get_paths()["tables_dir"] / "language_model_performance.csv"
    if not path.exists():
        logger.info("Skipping 18 (no per-language model eval yet)")
        return None
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["subset"], df["macro_f1"], color="#5a6ea5")
    ax.set_title("Model performance by language subset (macro F1)")
    ax.set_ylabel("Macro F1")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "18_language_specific_model_performance.png")


def fig_flat_vs_hierarchical():
    path = sc.get_paths()["tables_dir"] / "flat_vs_hierarchical.csv"
    if not path.exists():
        logger.info("Skipping 19 (flat-vs-hierarchical comparison not run yet)")
        return None
    df = pd.read_csv(path).dropna(subset=["macro_f1"])
    if df.empty:
        logger.info("Skipping 19 (no measured macro_f1 yet)")
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#a5533b", "#4b7f52"][:len(df)]
    ax.bar(df["approach"], df["macro_f1"], color=colors)
    ax.set_title("Flat vs hierarchical architecture (held-out macro F1)")
    ax.set_ylabel("Macro F1")
    return _save(fig, "19_flat_vs_hierarchical.png")


# --- 21-22 -----------------------------------------------------------------
def fig_app_version_trend(master):
    if "app_version" not in master.columns:
        logger.info("Skipping 21 (no app_version column)")
        return None
    d = master.dropna(subset=["app_version"]).copy()
    if d.empty:
        logger.info("Skipping 21 (app_version empty)")
        return None
    d["app_version"] = d["app_version"].astype(str)
    top = d["app_version"].value_counts().head(15).index
    d = d[d["app_version"].isin(top)]
    g = (d.assign(fail=d["failure_flag"] == "YES")
         .groupby("app_version")["fail"].mean().mul(100)
         .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(g.index, g.to_numpy(), color="#a5533b")
    ax.set_title("Failure rate by app version (top 15 versions) - "
                 "association only, not causation")
    ax.set_ylabel("Failure rate (%)")
    ax.tick_params(axis="x", rotation=60)
    return _save(fig, "21_app_version_trend.png")


def fig_potential_anomaly_timeline(master):
    ew_path = sc.get_paths()["tables_dir"] / "early_warning.csv"
    if not ew_path.exists():
        logger.info("Skipping 22 (no early-warning table)")
        return None
    ew = pd.read_csv(ew_path)
    d = master.dropna(subset=["review_date"]).copy()
    if d.empty:
        logger.info("Skipping 22 (no dates)")
        return None
    d["month"] = (pd.to_datetime(d["review_date"])
                  .dt.to_period("M").dt.to_timestamp())
    g = (d.assign(fail=d["failure_flag"] == "YES")
         .groupby(["bank_type", "month"])["fail"].mean().mul(100).unstack(0))
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in g.columns:
        ax.plot(g.index, g[col], marker="o", ms=3, label=col)
    for _, row in ew[ew["status"] == "Potential anomaly"].iterrows():
        ax.axvline(pd.Timestamp(row["current_month"]), color="red",
                   linestyle="--", alpha=0.5)
    ax.set_title("Failure-rate timeline with potential-anomaly markers")
    ax.set_ylabel("Failure rate (%)")
    ax.legend(title="Bank type")
    return _save(fig, "22_potential_anomaly_timeline.png")


# --- orchestrator ----------------------------------------------------------
def build_all_figures(master: pd.DataFrame) -> list[Path]:
    """Generate every s.56 figure. Individual failures never stop the run."""
    jobs = [
        lambda: fig_sentiment_distribution(master),
        lambda: fig_sentiment_over_time(master),
        lambda: fig_govt_vs_private_sentiment(master),
        lambda: fig_technical_vs_financial_risk(master),
        lambda: fig_risk_by_bank(master),
        lambda: fig_risk_by_language(master),
        lambda: fig_risk_by_rating(master),
        lambda: fig_emotion_distribution(master),
        lambda: fig_emotion_by_risk(master),
        lambda: fig_aspect_frequency(master),
        lambda: fig_aspect_sentiment(master),
        fig_pagerank_top_aspects,
        fig_pagerank_govt_vs_private,
        lambda: fig_response_latency_distribution(master),
        lambda: fig_response_latency_vs_rating(master),
        fig_model_benchmark,
        fig_confusion_matrices,
        fig_language_specific_performance,
        fig_flat_vs_hierarchical,
        lambda: fig_severity_distribution(master),
        lambda: fig_app_version_trend(master),
        lambda: fig_potential_anomaly_timeline(master),
    ]
    written: list[Path] = []
    for job in jobs:
        try:
            result = job()
            if result:
                written.append(result)
        except Exception as e:
            logger.warning("Figure job failed: %s", e)
    logger.info("Figures written: %d/%d", len(written), len(jobs))
    return written


if __name__ == "__main__":
    m = pd.read_parquet(sc.get_paths()["predictions_dir"] /
                        "master_predictions.parquet")
    build_all_figures(m)