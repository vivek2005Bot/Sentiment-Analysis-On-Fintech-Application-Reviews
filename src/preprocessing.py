"""Phase 4: conservative preprocessing for transformer NLP (plan section 8).

Produces `review_text_model` from `review_text` WITHOUT aggressive cleaning:
- unescapes HTML entities, strips malformed HTML tags, collapses whitespace;
- PRESERVES emojis, punctuation, case, Hinglish, Indian abbreviations
  (UPI/KYC/OTP/NEFT/IMPS...), transliterated and code-mixed content.
Raw text is never modified. Emoji metadata is captured, not deleted.
"""

from __future__ import annotations

import html as _html
import re

import pandas as pd

import src.config as sc
from src.utils import EMOJI_CHAR_CLASS, extract_emojis, setup_logging

logger = setup_logging("preprocessing")

_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_WS_RE = re.compile(r"\s+")


def clean_single(text) -> str:
    """Single-text model-text builder (also used by tests)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cfg = sc.get_config()["preprocessing"]
    t = str(text)
    if cfg["html_unescape"]:
        t = _html.unescape(t)
    if cfg["strip_html_tags"]:
        t = _TAG_RE.sub(" ", t)
    if cfg["collapse_whitespace"]:
        t = _WS_RE.sub(" ", t).strip()
    return t


def add_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add review_text_model + emoji/length metadata columns (vectorized)."""
    cfg = sc.get_config()["preprocessing"]
    out = df.copy()
    raw = out["review_text"].astype("string")
    out["review_text_raw"] = raw

    model = raw.map(clean_single)  # python-level per row, preserves semantics
    out["review_text_model"] = model

    out["text_length_chars"] = model.str.len().fillna(0).astype("int32")
    out["text_length_words"] = model.str.count(r"\S+").fillna(0).astype("int32")

    out["emoji_count"] = raw.fillna("").str.count(EMOJI_CHAR_CLASS).astype("int32")
    out["emoji_present"] = out["emoji_count"] > 0
    if cfg["keep_emojis"]:
        out["emoji_types"] = raw.map(
            lambda s: "".join(dict.fromkeys(extract_emojis(s)))[:24]
        )
    else:  # pragma: no cover - config-off path
        out["emoji_types"] = ""
    return out


def preserve_token_check(df: pd.DataFrame) -> dict[str, int]:
    """Count preserved banking tokens (plan section 8 sanity check)."""
    text = df["review_text_model"].astype("string").str.upper()
    return {
        tok: int(text.str.contains(rf"\b{re.escape(tok)}\b", na=False).sum())
        for tok in sc.get_config()["preprocessing"]["preserve_tokens"]
    }
