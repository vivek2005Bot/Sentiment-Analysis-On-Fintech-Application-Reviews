"""Shared utilities: logging, seeds, PII masking, emoji helpers, removal log.

Rules implemented here (see INDIAN_BANKING_NLP_IMPLEMENTATION_PLAN.md):
- Rule 5:  every row removal is logged (before/removed/after + reason).
- Rule 11: PII is masked before any review text reaches the dashboard.
- Rule 41: reusable PII-masking function (emails, phones, account/txn digits).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as sc


# ---------------------------------------------------------------------------
# Logging / reproducibility
# ---------------------------------------------------------------------------
def setup_logging(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    paths = sc.get_paths()
    fh = logging.FileHandler(paths["logs_dir"] / "pipeline.log", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(level)
    return logger


def set_seeds(seed: int | None = None) -> int:
    """Seed python/numpy/torch. Returns the seed actually used."""
    seed = int(seed if seed is not None else sc.get_seed())
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional dependency
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


# ---------------------------------------------------------------------------
# PII masking (plan section 41) - masks for DISPLAY, raw data untouched
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?91[\s\-]?)?\b[6-9]\d{9}\b")
LONG_DIGIT_RE = re.compile(r"\b\d{9,18}\b")
LONG_ALNUM_RE = re.compile(r"\b(?=[A-Z0-9]*\d)(?:[A-Z0-9]{8,})\b", re.IGNORECASE)


def mask_pii(text) -> str:
    """Mask obvious PII. Email a***@example.com; Phone ******4321;
    Account/txn ****1234. Never alters stored data - display only."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)

    def _email(m: re.Match) -> str:
        local, _, domain = m.group(0).partition("@")
        return (local[0] if local else "") + "***@" + domain

    def _phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "******" + digits[-4:]

    # US-style 555-013-4567 style numbers and other digit groups >= 10
    def _digits(m: re.Match) -> str:
        s = m.group(0)
        return "****" + s[-4:]

    t = EMAIL_RE.sub(_email, t)
    t = re.sub(r"(?:\+?\d[\d\s\-.]{8,}\d)", _phone, t)  # generic phone groups
    t = LONG_DIGIT_RE.sub(_digits, t)
    t = LONG_ALNUM_RE.sub(lambda m: m.group(0)[:2] + "****" + m.group(0)[-2:], t)
    return t


# ---------------------------------------------------------------------------
# Emoji / artifact helpers (plan section 28: never delete emojis silently)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import emoji as _emoji_lib

    def extract_emojis(text: str) -> list[str]:
        try:
            return [ch for ch in str(text) if ch in _emoji_lib.EMOJI_DATA]
        except AttributeError:
            return [ch for ch in str(text) if _emoji_lib.is_emoji(ch)]

except Exception:  # pragma: no cover
    _EMOJI_RANGE_RE = re.compile(
        "[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\ufe0f\u2190-\u21ff]"
    )

    def extract_emojis(text: str) -> list[str]:
        return _EMOJI_RANGE_RE.findall(str(text))


EMOJI_CHAR_CLASS = "[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\ufe0f]"
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
HTML_ENTITY_RE = re.compile(r"&[A-Za-z#0-9]+;")


def count_emojis(text: str) -> int:
    return len(extract_emojis(text))


def has_html_artifacts(text: str) -> bool:
    return bool(HTML_TAG_RE.search(str(text))) or bool(HTML_ENTITY_RE.search(str(text)))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def sha1_short(*parts) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# JSON / misc IO
# ---------------------------------------------------------------------------
def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Removal logging (plan section 7 / Rule 5: never silently drop rows)
# ---------------------------------------------------------------------------
class RemovalLog:
    """Collects row-removal events and appends them to a CSV audit trail.

    Columns: stage, reason, source, rows_before, rows_removed, rows_after,
    timestamp. The log is the single source of truth for data-loss audits.
    """

    COLUMNS = [
        "timestamp",
        "stage",
        "reason",
        "source",
        "rows_before",
        "rows_removed",
        "rows_after",
    ]

    def __init__(self, path: Path | None = None):
        paths = sc.get_paths()
        self.path = Path(path) if path else paths["processed_dir"] / "removal_log.csv"
        self.rows: list[dict] = []

    def add(
        self,
        stage: str,
        reason: str,
        before: int,
        removed: int,
        after: int,
        source: str = "",
    ) -> None:
        if removed != before - after:  # consistency guard
            removed = before - after
        self.rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "reason": reason,
                "source": source,
                "rows_before": int(before),
                "rows_removed": int(removed),
                "rows_after": int(after),
            }
        )

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(self.rows, columns=self.COLUMNS)
        if self.path.exists():
            old = pd.read_csv(self.path)
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(self.path, index=False)
        return self.path

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLUMNS)
