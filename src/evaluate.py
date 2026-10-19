"""Held-out evaluation + language subsets + flat-vs-hierarchical (s.24-25).

Loads a trained model directory (models/<task>/<model>/model), evaluates on a
held-out test split of the annotation set, reports per-language-group metrics,
and builds confusion-matrix figures under reports/figures/.

Flat vs hierarchical (plan s.24): the hierarchical pipeline = failure gate
(binary) + risk head (2-class, failure rows only). The flat baseline = single
4-class model (NORMAL / OPERATIONAL_FRICTION / TECHNICAL_RISK /
FINANCIAL_RISK) trained from the same labels on the same test rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as sc
from src import model_registry as mr
from src.utils import setup_logging

logger = setup_logging("evaluate")

FLAT_LABELS = ["NORMAL", "OPERATIONAL_FRICTION", "TECHNICAL_RISK", "FINANCIAL_RISK"]


def flat_label(row) -> str:
    """Map hierarchical labels to the flat 4-class baseline (s.24)."""
    if str(row.get("failure_flag")) == "YES":
        if row.get("primary_risk") == "TECHNICAL_PERFORMANCE_RISK":
            return "TECHNICAL_RISK"
        if row.get("primary_risk") == "FINANCIAL_SECURITY_RISK":
            return "FINANCIAL_RISK"
        return "OPERATIONAL_FRICTION"
    if str(row.get("operational_friction")) == "YES":
        return "OPERATIONAL_FRICTION"
    return "NORMAL"


def predict_probs(model_dir, texts: list[str], device: str | None = None):
    """Batch probability predictions from a saved model directory."""
    import json

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or mr.get_device()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    probs = []
    batch = int(sc.get_model_config()["inference"]["batch_size"])
    maxlen = int(sc.get_model_config()["training"]["max_length"])
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                texts[i:i + batch],
                truncation=True,
                padding=True,
                max_length=maxlen,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    arr = np.vstack(probs)
    with open(f"{model_dir}/label_map.json") as fh:
        labels = json.load(fh)["labels"]
    return arr, labels


def evaluate_language_subsets(
    df: pd.DataFrame, y: np.ndarray, pred: np.ndarray
) -> pd.DataFrame:
    """Per-language-group precision/recall/macro-F1 (plan s.25)."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    candidates = {
        "English": ("language", lambda s: s == "English"),
        "Hinglish": ("language", lambda s: s == "Hinglish"),
        "Hindi": ("language", lambda s: s == "Hindi"),
        "Tamil": ("language", lambda s: s.isin(["Tamil", "Tanglish"])),
        "code_mixed": ("is_code_mixed", lambda s: s.eq(True)),
        "non_code_mixed": ("is_code_mixed", lambda s: s.eq(False)),
        "native_script": ("script", lambda s: s != "Latin"),
        "transliterated": ("is_transliterated", lambda s: s.eq(True)),
    }
    groups = {}
    for name, (col, cond) in candidates.items():
        if col in df.columns:  # tolerate frames lacking some metadata
            mask = cond(df[col])
            groups[name] = (
                mask.fillna(False) if hasattr(mask, "fillna") else mask
            )
    rows = []
    for name, mask in groups.items():
        mask = mask.fillna(False).to_numpy()
        if mask.sum() < 20:
            continue
        rows.append(
            {
                "subset": name,
                "n": int(mask.sum()),
                "accuracy": float((pred[mask] == y[mask]).mean()),
                "macro_f1": round(
                    float(
                        f1_score(y[mask], pred[mask], average="macro", zero_division=0)
                    ),
                    4,
                ),
                "precision_macro": round(
                    float(
                        precision_score(
                            y[mask], pred[mask], average="macro", zero_division=0
                        )
                    ),
                    4,
                ),
                "recall_macro": round(
                    float(
                        recall_score(
                            y[mask], pred[mask], average="macro", zero_division=0
                        )
                    ),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows)


def confusion_figure(y, pred, labels: list[str], title: str, fname: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out = sc.get_paths()["figures_dir"] / fname
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def flat_vs_hierarchical(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare flat 4-class vs hierarchical heads on the SAME test rows.

    Improvement is claimed only from measured held-out metrics (Rule 9).
    The flat head is a distinct distilbert fine-tune (task 'flat');
    hierarchical = failure gate + risk head scores from the benchmark log.
    """
    from sklearn.model_selection import train_test_split

    paths = sc.get_paths()
    if df is None:
        df = pd.read_csv(paths["annotations_dir"] / "annotation_sample_silver.csv")
    df = df[df["review_text_model"].notna()].copy()
    df["flat_label"] = df.apply(flat_label, axis=1)
    df = df[df["flat_label"].isin(FLAT_LABELS)]
    if len(df) < 100:
        logger.warning("flat_vs_hierarchical: too few labelled rows")
        return pd.DataFrame()
    _, test = train_test_split(
        df, test_size=0.2, random_state=sc.get_seed(), stratify=df["flat_label"]
    )
    rows = []

    flat_dir = mr.model_dir("flat", "distilbert-base-uncased") / "model"
    if (flat_dir / "config.json").exists():
        proba, labels = predict_probs(flat_dir, list(test["review_text_model"]))
        pred = proba.argmax(axis=1)
        y = test["flat_label"].map({l: i for i, l in enumerate(labels)}).to_numpy()
        from sklearn.metrics import f1_score, recall_score

        rows.append(
            {
                "approach": "flat_4class",
                "n_test": len(test),
                "macro_f1": round(
                    float(f1_score(y, pred, average="macro", zero_division=0)), 4
                ),
                "financial_recall": round(
                    float(
                        recall_score(
                            y,
                            pred,
                            average=None,
                            labels=[labels.index("FINANCIAL_RISK")],
                            zero_division=0,
                        )[0]
                        if "FINANCIAL_RISK" in labels
                        else np.nan
                    ),
                    4,
                ),
            }
        )

    # Hierarchical side: measured end-to-end scores only. The pipeline-level
    # failure+risk heads have no trained transformer artifacts yet, so there is
    # nothing measured to report here (Rule 9: no fabricated rows). When
    # failure/risk heads are trained, extend this block with their measured
    # held-out scores on these SAME test rows.
    if not rows:
        logger.warning(
            "flat_vs_hierarchical: no trained flat head found - "
            "table left untouched"
        )
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables_dir"] / "flat_vs_hierarchical.csv", index=False)
    return out


if __name__ == "__main__":
    print(flat_vs_hierarchical())
