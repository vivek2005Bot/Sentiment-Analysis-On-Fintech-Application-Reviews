"""Transformer model registry (plan sections 19-23).

Responsibilities: enabled-model discovery per task, device selection
(MPS on Apple Silicon), checkpoint validation (sequence-classification head),
experiment tracking JSON/CSV under reports/model_results/ (plan s.46), and
artifact paths under models/<task>/<model_slug>/.

Training never runs inside Streamlit (plan Rule 12) - this module is used by
train.py / evaluate.py / inference only.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

import src.config as sc
from src.utils import save_json, set_seeds, setup_logging

logger = setup_logging("model_registry")

LABEL_MAPS = {
    "sentiment": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
    "failure": ["NO", "YES"],
    "risk": ["TECHNICAL_PERFORMANCE_RISK", "FINANCIAL_SECURITY_RISK"],
    "emotion": [
        "ANGER",
        "FRUSTRATION",
        "FEAR",
        "SADNESS",
        "DISAPPOINTMENT",
        "JOY",
        "TRUST",
        "CONFUSION",
        "NEUTRAL",
        "OTHER",
    ],
    # Flat 4-class baseline for the s.24 flat-vs-hierarchical comparison.
    # Labels derive from hierarchical columns (see evaluate.flat_label)
    # so the flat head trains on the SAME rows as the gate+risk heads.
    "flat": ["NORMAL", "OPERATIONAL_FRICTION", "TECHNICAL_RISK", "FINANCIAL_RISK"],
}


def get_device() -> str:
    cfg = sc.get_model_config()["inference"]["device"]
    if cfg == "auto":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return cfg


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def enabled_models(task: str | None = None) -> list[dict]:
    cfg = sc.get_model_config()
    out = []
    for m in cfg["benchmark"]["models"]:
        if not m.get("enabled"):
            continue
        if task and task not in m.get("task_fit", []):
            continue
        out.append(m)
    return out


def model_dir(task: str, model_name: str) -> Path:
    paths = sc.get_paths()
    d = paths["models_dir"] / task / slugify(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def validate_checkpoint(model_name: str) -> bool:
    """Import + instantiate config/tokenizer to verify compatibility."""
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        return True
    except Exception as e:
        logger.error("Checkpoint %s failed validation: %s", model_name, e)
        return False


def git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(sc.PROJECT_ROOT),
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def start_experiment(model_name: str, task: str, extra: dict | None = None) -> dict:
    seed = set_seeds()
    paths = sc.get_paths()
    exp = {
        "experiment_id": (
            f"{task}_{slugify(model_name)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
        "model": model_name,
        "task": task,
        "dataset_version": "master_v1",
        "seed": seed,
        "device": get_device(),
        "git_commit": git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        exp.update(extra)
    save_json(exp, paths["model_results_dir"] / f"{exp['experiment_id']}.meta.json")
    return exp


def record_metrics(experiment: dict, metrics: dict) -> Path:
    paths = sc.get_paths()
    exp = {
        **experiment,
        **metrics,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_json = paths["model_results_dir"] / f"{exp['experiment_id']}.json"
    save_json(exp, out_json)

    csv_path = paths["model_results_dir"] / "benchmark_log.csv"
    flat = {
        "experiment_id": exp["experiment_id"],
        "model": exp["model"],
        "task": exp["task"],
        "device": exp.get("device", ""),
        "seed": exp.get("seed", ""),
    }
    flat.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})
    df = pd.DataFrame([flat])
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    logger.info("Metrics recorded -> %s", out_json)
    return out_json


def load_experiment_results() -> pd.DataFrame:
    paths = sc.get_paths()
    csv_path = paths["model_results_dir"] / "benchmark_log.csv"
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
