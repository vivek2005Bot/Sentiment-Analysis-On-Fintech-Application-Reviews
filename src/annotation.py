"""Phase 6 (plan s.10): stratified annotation sample + silver labels.

Two artifacts:
1. annotation_sample.csv  - stratified by bank x rating x language group x
   period, with EMPTY human label columns (ready for annotation). This is the
   only dataset allowed for final model evaluation.
2. annotation_sample_silver.csv - the same reviews with deterministic
   rule-based 'silver' labels (clearly marked label_source). Silver labels
   bootstrap early experiments only and are NEVER reported as human
   annotations (plan Rules 9, 15).
"""

from __future__ import annotations

import pandas as pd

import src.config as sc
from src import (
    aspect_engine,
    emotion_engine,
    failure_detector,
    language_detector,
    preprocessing,
    risk_engine,
    sentiment_engine,
)
from src.utils import save_json, setup_logging

logger = setup_logging("annotation")

HUMAN_LABEL_COLUMNS = [
    "sentiment",
    "failure_flag",
    "operational_friction",
    "critical_failure",
    "primary_risk",
    "secondary_risk",
    "risk_overlap",
    "emotion",
    "severity",
    "aspect",
    "aspect_sentiment",
    "annotation_confidence",
    "annotator_id",
    "annotation_version",
]


def _lang_group(lang) -> str:
    if lang in ("English",):
        return "English"
    if lang in ("Hinglish", "Tanglish"):
        return "Hinglish_code_mixed"
    if lang in (
        "Hindi",
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
        "Bengali",
        "Marathi",
        "Gujarati",
        "Punjabi",
    ):
        return "Indic_script"
    return "Other"


def build_annotation_sample(
    master: pd.DataFrame,
    target: int | None = None,
    out_dir=None,
    force: bool = False,
) -> pd.DataFrame:
    """Stratified sample with empty human label columns.

    ``out_dir`` redirects the outputs (used by demo/smoke runs so a 5-row
    sample can never overwrite the real corpus sample). If an existing sample
    would be replaced by a smaller one, a loud warning is logged unless
    ``force=True`` -- this makes accidental clobbering visible (plan Rule 15).
    """
    cfg = sc.get_config()["annotation"]
    target = int(target or cfg["target_sample"])
    seed = sc.get_seed()
    df = master[master["is_empty_text"].eq(False)].copy()
    if "review_text_model" not in df.columns:
        df = preprocessing.add_text_fields(df)
    if "language" not in df.columns:
        df = language_detector.detect_language_frame(df)
    df["lang_group"] = df["language"].map(_lang_group)
    df["month_period"] = (
        pd.to_datetime(df["review_date"], errors="coerce").dt.to_period("M").astype(str)
    )

    parts = []
    for _, sub in df.groupby(
        ["bank", "rating", "lang_group"], dropna=False, observed=True
    ):
        take = max(cfg["min_per_bank"] // 6, int(round(target * len(sub) / len(df))))
        n = min(take, len(sub))
        if n > 0:
            parts.append(sub.sample(n=n, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)
    # trim to target if oversampled
    if len(sample) > target * 1.3:
        sample = sample.sample(n=target, random_state=seed)

    for col in HUMAN_LABEL_COLUMNS:
        sample[col] = pd.NA
    sample["annotation_version"] = cfg["annotation_version"]

    out_dir = out_dir or sc.get_paths()["annotations_dir"]
    out = out_dir / "annotation_sample.csv"
    if out.exists():
        try:
            existing_rows = sum(1 for _ in open(out, encoding="utf-8")) - 1
        except OSError:
            existing_rows = -1
        if existing_rows > len(sample) and not force:
            logger.warning(
                "OVERWRITE GUARD: existing human sample has %d rows but new "
                "sample has only %d (%s). Writing anyway is dangerous; pass "
                "force=True or out_dir to acknowledge. Proceeding with write.",
                existing_rows, len(sample), out,
            )
        elif existing_rows != len(sample):
            logger.info(
                "Replacing human sample: %d -> %d rows (%s)",
                existing_rows, len(sample), out,
            )
    sample.to_csv(out, index=False)
    meta = {
        "rows": int(len(sample)),
        "target": target,
        "bank_coverage": int(sample["bank"].nunique()),
        "rating_coverage": sorted({int(r) for r in sample["rating"].dropna().unique()}),
        "language_group_counts": sample["lang_group"].value_counts().to_dict(),
        "generated_at": pd.Timestamp.now().isoformat(),
        "note": "Empty label columns await HUMAN annotation.",
    }
    save_json(meta, out_dir / "annotation_sample_meta.json")
    logger.info("Annotation sample: %d rows -> %s", len(sample), out)
    return sample


def add_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic rule-based labels (bootstrap only, never 'human')."""
    # Drop any stale engine output columns so re-runs never create duplicates.
    stale = [c for c in df.columns if c in {
        "review_text_raw", "review_text_model", "text_length_chars",
        "text_length_words", "emoji_count", "emoji_present", "emoji_types",
        "language", "script", "is_code_mixed", "is_transliterated",
        "language_confidence", "hinglish_ratio", "sentiment",
        "sentiment_score", "sentiment_confidence", "sentiment_model_source",
        "failure_flag", "operational_friction", "critical_failure",
        "failure_confidence", "failure_signals", "friction_signals",
        "failure_model_source", "primary_risk", "primary_risk_confidence",
        "secondary_risk", "risk_overlap", "severity", "risk_model_source",
        "emotion", "emotion_all", "emotion_confidence", "emotion_model_source",
        "aspects", "aspect_count", "aspect_sentiment", "label_source",
        "rating_sentiment_mismatch"}]
    df = df.drop(columns=stale, errors="ignore")
    out = preprocessing.add_text_fields(df)
    out = language_detector.detect_language_frame(out)
    out = sentiment_engine.score_sentiment_frame(out)
    out = failure_detector.detect_failure_frame(out)
    out = risk_engine.classify_risk_frame(out)
    out = emotion_engine.detect_emotion_frame(out)
    out = aspect_engine.detect_aspects_frame(out)
    out["label_source"] = sc.get_config()["annotation"]["silver_label_version"]
    return out


def build_silver_annotation_set(out_dir=None) -> pd.DataFrame:
    paths = sc.get_paths()
    ann_dir = out_dir or paths["annotations_dir"]
    sample_csv = ann_dir / "annotation_sample.csv"
    sample = (
        pd.read_csv(sample_csv)
        if sample_csv.exists()
        else build_annotation_sample(
            pd.read_parquet(paths["master_parquet"]), out_dir=ann_dir
        )
    )
    silver = add_silver_labels(sample)
    out = ann_dir / "annotation_sample_silver.csv"
    silver.to_csv(out, index=False)
    logger.info("Silver annotation set: %d rows -> %s", len(silver), out)
    return silver


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase 6 annotation sample builder")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Run on data/samples/demo_reviews.csv into data/samples/annotations_demo "
        "(never touches the real corpus sample).",
    )
    ap.add_argument("--force", action="store_true",
                    help="Allow replacing a larger existing sample (use with care).")
    args = ap.parse_args()

    paths = sc.get_paths()
    if args.demo:
        demo = pd.read_csv(paths["samples_dir"] / "demo_reviews.csv")
        if "bank_type" not in demo.columns:
            demo["bank_type"] = "GOVERNMENT"
        demo_dir = paths["samples_dir"] / "annotations_demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DEMO MODE: %d demo rows -> %s", len(demo), demo_dir)
        build_annotation_sample(
            demo, target=len(demo), out_dir=demo_dir, force=True
        )
        build_silver_annotation_set(out_dir=demo_dir)
    else:
        master = pd.read_parquet(paths["master_parquet"])
        build_annotation_sample(master, force=args.force)
        build_silver_annotation_set()
