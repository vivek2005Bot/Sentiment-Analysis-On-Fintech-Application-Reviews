"""Prediction/report exporters (plan s.54 expected output files).

Splits the master prediction table into the per-task parquet files required by
the plan, plus the aspect co-occurrence edge list, the consolidated
pagerank_by_bank table, and the model benchmark table (s.55 Table 3).
Written by src.pipeline after the engines have run.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd

import src.config as sc
from src.utils import setup_logging

logger = setup_logging("export")

# task -> columns exported to its own parquet (plan s.54)
TASK_COLUMNS = {
    "sentiment": ["review_id", "bank", "bank_type", "sentiment",
                  "sentiment_score", "sentiment_confidence",
                  "sentiment_model_source", "rating",
                  "rating_sentiment_mismatch"],
    "failure": ["review_id", "bank", "bank_type", "failure_flag",
                "failure_confidence", "operational_friction",
                "critical_failure", "failure_signals", "friction_signals",
                "failure_model_source"],
    "risk": ["review_id", "bank", "bank_type", "failure_flag", "primary_risk",
             "primary_risk_confidence", "secondary_risk", "risk_overlap",
             "severity", "risk_model_source"],
    "emotion": ["review_id", "bank", "bank_type", "emotion", "emotion_all",
                "emotion_confidence", "emotion_model_source"],
    "aspect": ["review_id", "bank", "bank_type", "aspects", "aspect_count",
               "aspect_sentiment"],
}


def write_task_predictions(master: pd.DataFrame) -> dict[str, int]:
    """Write data/predictions/<task>_predictions.parquet for each task."""
    out_dir = sc.get_paths()["predictions_dir"]
    written = {}
    for task, cols in TASK_COLUMNS.items():
        present = [c for c in cols if c in master.columns]
        missing = [c for c in cols if c not in master.columns]
        if missing:
            logger.warning("Task %s missing columns (skipped): %s",
                           task, missing)
        sub = master[present]
        path = out_dir / f"{task}_predictions.parquet"
        sub.to_parquet(path, index=False)
        written[task] = len(sub)
        logger.info("Wrote %s (%d rows)", path.name, len(sub))
    return written


def write_aspect_edges(master: pd.DataFrame) -> pd.DataFrame:
    """Aspect co-occurrence edge list -> aspect_edges.parquet (plan s.54).

    Undirected pairs are normalized (a, b) with a < b so the edge table is
    canonical and directly consumable by the PageRank engine.
    """
    pair_counts: Counter = Counter()
    single_counts: Counter = Counter()
    for aspects in master["aspects"].dropna():
        items = sorted({a for a in str(aspects).split(",") if a})
        for a in items:
            single_counts[a] += 1
        for a, b in combinations(items, 2):
            pair_counts[(a, b)] += 1

    edges = pd.DataFrame(
        [{"aspect_a": a, "aspect_b": b, "weight": w}
         for (a, b), w in pair_counts.items()]
    ).sort_values("weight", ascending=False).reset_index(drop=True)
    edges["mention_a"] = edges["aspect_a"].map(single_counts)
    edges["mention_b"] = edges["aspect_b"].map(single_counts)

    path = sc.get_paths()["predictions_dir"] / "aspect_edges.parquet"
    edges.to_parquet(path, index=False)
    logger.info("Wrote %s (%d edges)", path.name, len(edges))
    return edges


def write_pagerank_by_bank() -> pd.DataFrame:
    """Consolidate per-bank PageRank CSVs into pagerank_by_bank.csv (s.54)."""
    tables_dir = sc.get_paths()["tables_dir"]
    frames = []
    for path in sorted(tables_dir.glob("pagerank_bank_*.csv")):
        bank = path.stem.replace("pagerank_bank_", "").replace("_", " ")
        df = pd.read_csv(path)
        df.insert(0, "bank", bank)
        frames.append(df)
    if not frames:
        logger.warning("No per-bank PageRank tables found to consolidate")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["rank_in_bank"] = (out.groupby("bank")["pagerank"]
                           .rank(ascending=False, method="min").astype(int))
    path = sc.get_paths()["tables_dir"] / "pagerank_by_bank.csv"
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path.name, len(out))
    return out


def write_model_benchmark() -> pd.DataFrame:
    """Table 3 (plan s.55): Model | Task | Accuracy | Macro F1 | ..."""
    paths = sc.get_paths()
    log = paths["model_results_dir"] / "benchmark_log.csv"
    if not log.exists():
        logger.warning("benchmark_log.csv not found - no benchmark table yet")
        return pd.DataFrame()
    df = pd.read_csv(log)
    cols = {
        "model": "model", "task": "task", "accuracy": "accuracy",
        "macro_f1": "macro_f1", "weighted_f1": "weighted_f1",
        "precision_macro": "precision", "recall_macro": "recall",
        "inference_time_s": "inference_time_s", "split": "split",
        "train_size": "train_size", "test_size": "test_size",
        "device": "device", "seed": "seed",
    }
    present = {k: v for k, v in cols.items() if k in df.columns}
    out = df[list(present)].rename(columns=present)
    out = out.sort_values(["task", "macro_f1"], ascending=[True, False])
    path = paths["tables_dir"] / "model_benchmark.csv"
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path.name, len(out))
    return out


def run_exports(master: pd.DataFrame) -> None:
    write_task_predictions(master)
    write_aspect_edges(master)
    write_pagerank_by_bank()
    write_model_benchmark()
    logger.info("All s.54 exports complete")