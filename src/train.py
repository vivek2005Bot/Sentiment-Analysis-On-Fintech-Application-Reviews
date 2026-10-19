"""Fine-tune a transformer on the annotation set (plan sections 19-23, 43).

Usage:
  .venv/bin/python -m src.train --task sentiment --model distilbert-base-uncased [--smoke]
  .venv/bin/python -m src.train --task risk --split cross_bank

Data: data/annotations/annotation_sample_silver.csv by default (use --data
for human-annotated files). Silver labels are bootstrap labels - benchmarks
built on them are provisional until human labels exist (documented).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import src.config as sc
from src import model_registry as mr
from src.utils import setup_logging

logger = setup_logging("train")


def load_task_data(task: str, data_path=None) -> pd.DataFrame:
    paths = sc.get_paths()
    path = data_path or (paths["annotations_dir"] / "annotation_sample_silver.csv")
    df = pd.read_csv(path)
    if task == "flat" and "flat_label" not in df.columns:
        from src.evaluate import flat_label as _flat_label

        df["flat_label"] = df.apply(_flat_label, axis=1)
        logger.info("Derived flat_label: %s", df["flat_label"].value_counts().to_dict())
    cfg = sc.get_model_config()["tasks"][task]
    text_col = cfg["text_column"]
    label_col = cfg["label_column"]
    if "filter_column" in cfg:
        mask = df[cfg["filter_column"]].astype(str) == str(cfg["filter_value"])
        df = df[mask].copy()
    df = df[df[text_col].notna() & df[label_col].notna()].copy()
    labels = mr.LABEL_MAPS[task]
    df = df[df[label_col].astype(str).str.upper().isin([lbl.upper() for lbl in labels])]
    df["label_id"] = (
        df[label_col]
        .astype(str)
        .str.upper()
        .map({l.upper(): i for i, l in enumerate(labels)})
    )
    logger.info(
        "Task %s: %d rows from %s | labels: %s",
        task,
        len(df),
        path.name,
        df[label_col].value_counts().to_dict(),
    )
    return df


def make_splits(df: pd.DataFrame, task: str, split: str = "random"):
    """Random (default), time, or cross-bank splits - plan section 26."""
    from sklearn.model_selection import train_test_split

    ev = sc.get_model_config()["evaluation"]
    seed = sc.get_seed()
    strat = df["label_id"] if df["label_id"].value_counts().min() >= 2 else None

    if split == "time":
        df = df.sort_values("review_date")
        n_train = int(0.7 * len(df))
        n_val = int(0.15 * len(df))
        train, rest = df.iloc[:n_train], df.iloc[n_train:]
        val, test = rest.iloc[:n_val], rest.iloc[n_val:]
        return train, val, test

    if split == "cross_bank":
        banks = sorted(df["bank"].unique())
        test_bank = banks[int(seed) % len(banks)]
        test = df[df["bank"] == test_bank]
        rest = df[df["bank"] != test_bank]
        train, val = train_test_split(
            rest,
            test_size=0.18,
            random_state=seed,
            stratify=(
                rest["label_id"] if rest["label_id"].value_counts().min() >= 2 else None
            ),
        )
        logger.info("Cross-bank split: held-out bank = %s", test_bank)
        return train, val, test

    train, rest = train_test_split(
        df,
        test_size=ev["val_size"] + ev["test_size"],
        random_state=seed,
        stratify=strat,
    )
    strat_rest = (
        rest["label_id"] if rest["label_id"].value_counts().min() >= 2 else None
    )
    val, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=strat_rest
    )
    return train, val, test


def train_one(
    task: str,
    model_name: str,
    smoke: bool = False,
    split: str = "random",
    data_path=None,
) -> dict:
    import json

    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tcfg = sc.get_model_config()["tasks"][task]
    trcfg = sc.get_model_config()["training"]
    labels = mr.LABEL_MAPS[task]
    exp = mr.start_experiment(
        model_name,
        task,
        extra={
            "split": split,
            "smoke": smoke,
            "hyperparameters": {
                k: trcfg[k]
                for k in (
                    "max_length",
                    "learning_rate",
                    "epochs",
                    "train_batch_size",
                    "weight_decay",
                    "warmup_ratio",
                )
            },
        },
    )
    df = load_task_data(task, data_path)
    train, val, test = make_splits(df, task, split)
    if smoke:
        train = train.head(trcfg["smoke_run"]["max_train_samples"])
        val = val.head(trcfg["smoke_run"]["max_eval_samples"])
        test = test.head(trcfg["smoke_run"]["max_eval_samples"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class ReviewsDataset(Dataset):
        def __init__(self, frame):
            self.enc = tokenizer(
                list(frame[tcfg["text_column"]].astype(str)),
                truncation=True,
                padding="max_length",
                max_length=trcfg["max_length"],
            )
            self.y = list(frame["label_id"])

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i])
            return item

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(
                y, pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        }

    trargs = TrainingArguments(
        output_dir=str(mr.model_dir(task, model_name) / "checkpoints"),
        num_train_epochs=(trcfg["smoke_run"]["epochs"] if smoke else trcfg["epochs"]),
        per_device_train_batch_size=trcfg["train_batch_size"],
        per_device_eval_batch_size=trcfg["eval_batch_size"],
        learning_rate=trcfg["learning_rate"],
        weight_decay=trcfg["weight_decay"],
        warmup_steps=int(trcfg["warmup_ratio"] * max(1,
            (len(train) // (trcfg["train_batch_size"] *
                            trcfg["gradient_accumulation_steps"])) *
            (trcfg["smoke_run"]["epochs"] if smoke else trcfg["epochs"]))),
        gradient_accumulation_steps=trcfg["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="no" if smoke else "epoch",
        load_best_model_at_end=not smoke,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        seed=exp["seed"],
        report_to=[],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)
    )
    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=trargs,
        train_dataset=ReviewsDataset(train),
        eval_dataset=ReviewsDataset(val),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    training_time = time.time() - t0

    t1 = time.time()
    test_out = trainer.evaluate(ReviewsDataset(test))
    inference_time = time.time() - t1

    metrics = {
        "split": split,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "accuracy": round(float(test_out["eval_accuracy"]), 4),
        "macro_f1": round(float(test_out["eval_macro_f1"]), 4),
        "weighted_f1": round(float(test_out["eval_weighted_f1"]), 4),
        "precision_macro": round(float(test_out["eval_precision_macro"]), 4),
        "recall_macro": round(float(test_out["eval_recall_macro"]), 4),
        "training_time_s": round(training_time, 1),
        "inference_time_s": round(inference_time, 2),
        "model_source": "transformer",
    }
    mr.record_metrics(exp, metrics)

    # Held-out prediction dumps + language-subset performance (s.25/s.56).
    # Both smoke and full runs export so figures 17/18 can be built.
    try:
        eval_out = trainer.predict(ReviewsDataset(test))
        pred = np.argmax(eval_out.predictions, axis=-1)
        export_eval_artifacts(task, test, np.asarray(test["label_id"]), pred)
    except Exception as e:
        logger.warning("Eval artifact export skipped: %s", e)

    # Persist the model in both smoke and full runs: evaluate.flat_vs_hierarchical
    # needs the saved flat head on disk. Smoke runs still skip checkpoints
    # (handled above) but must export the final model + label map.
    save_dir = mr.model_dir(task, model_name)
    model_out = save_dir / "model"
    trainer.save_model(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    with open(model_out / "label_map.json", "w") as fh:
        json.dump({"labels": labels}, fh)
    # Back-compat: evaluate.predict_probs looks up <model_dir>/label_map.json
    # where model_dir already ends in /model, so keep a copy at the parent too.
    with open(save_dir / "label_map.json", "w") as fh:
        json.dump({"labels": labels}, fh)
    logger.info(
        "[%s/%s] macro_f1=%.4f (smoke=%s, split=%s)",
        task,
        model_name,
        metrics["macro_f1"],
        smoke,
        split,
    )
    return metrics


def export_eval_artifacts(task: str, test_df: pd.DataFrame,
                          y: np.ndarray, pred: np.ndarray) -> None:
    """Persist held-out predictions + per-language performance (s.25/s.56).

    Feeds figures 17 (confusion matrices) and 18 (language-specific model
    performance), and the flat-vs-hierarchical comparison input.
    """
    paths = sc.get_paths()
    dump = pd.DataFrame({
        "task": task,
        "review_id": test_df["review_id"].to_numpy()
        if "review_id" in test_df.columns else np.arange(len(test_df)),
        "y_true": y,
        "y_pred": pred,
    })
    dump_path = paths["predictions_dir"] / f"{task}_predictions_eval.parquet"
    dump.to_parquet(dump_path, index=False)

    from src import evaluate

    lang = evaluate.evaluate_language_subsets(
        test_df.reset_index(drop=True), y, pred)
    if lang.empty:
        logger.info("No language subsets large enough for %s", task)
        return
    lang.insert(0, "task", task)
    out = paths["tables_dir"] / "language_model_performance.csv"
    if out.exists():
        old = pd.read_csv(out)
        old = old[old["task"] != task]  # replace this task's previous rows
        lang = pd.concat([old, lang], ignore_index=True)
    lang.to_csv(out, index=False)
    logger.info("Eval artifacts for %s -> %s", task, dump_path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(mr.LABEL_MAPS.keys()))
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--split", default="random", choices=["random", "time", "cross_bank"]
    )
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    train_one(
        args.task, args.model, smoke=args.smoke, split=args.split, data_path=args.data
    )


if __name__ == "__main__":
    main()
