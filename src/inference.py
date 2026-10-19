"""Full-corpus transformer inference (plan sections 18, 40, 42, 54).

Loads a trained artifact, runs batched prediction over master_predictions,
writes per-task prediction parquet files, and merges into the master
predictions with model_source='transformer'. Heavy training stays offline;
this script is for offline batch inference (never inside Streamlit).
"""

from __future__ import annotations

import argparse

import pandas as pd

import src.config as sc
from src import model_registry as mr
from src.evaluate import predict_probs
from src.utils import setup_logging

logger = setup_logging("inference")


def run_inference(task: str, model_dir=None, limit: int | None = None) -> pd.DataFrame:
    """Predict <task> for every master row and write predictions parquet."""
    paths = sc.get_paths()
    model_dir = model_dir or (mr.model_dir(task, "distilbert-base-uncased") / "model")
    master = pd.read_parquet(paths["predictions_dir"] / "master_predictions.parquet")
    if limit:
        master = master.head(limit)
    df = master[master["review_text_model"].notna()].copy()
    logger.info("Inference %s: %d rows | device=%s", task, len(df), mr.get_device())
    proba, labels = predict_probs(model_dir, list(df["review_text_model"]))
    pred = [labels[i] for i in proba.argmax(axis=1)]
    conf = proba.max(axis=1).round(4)

    out = pd.DataFrame(
        {
            "review_id": df["review_id"].to_numpy(),
            f"{task}_pred": pred,
            f"{task}_confidence": conf,
            "model_source": "transformer",
        }
    )
    out_path = paths["predictions_dir"] / f"{task}_predictions_transformer.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(mr.LABEL_MAPS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_inference(args.task, limit=args.limit)
