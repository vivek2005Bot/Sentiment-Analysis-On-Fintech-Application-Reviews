"""Aspect engine (plan sections 17-18): aspect detection + aspect sentiment.

Aspect taxonomy v1 from the plan (APP_PERFORMANCE ... OTHER), matched with
versioned lexicon patterns including Hinglish variants. Each detected aspect
carries its own local sentiment (positive/negative/neutral context windows),
so a review like "app is good but UPI is terrible" yields per-aspect labels.
"""

from __future__ import annotations

import re

import pandas as pd

from src.lexicons import compile_any
from src.utils import setup_logging

logger = setup_logging("aspect_engine")

TAXONOMY_VERSION = "banking_aspects_v1"

ASPECT_PATTERNS: dict[str, list[str]] = {
    "APP_PERFORMANCE": [
        r"app (?:is )?(?:slow|lag|hang|freez\w*)",
        r"performance",
        r"speed",
        r"slow",
        r"laggy",
        r"hanging",
        r"freez\w*",
        r"optimize",
    ],
    "UI_UX": [
        r"\bui\b",
        r"\bux\b",
        r"user interface",
        r"user experience",
        r"design",
        r"look and feel",
        r"theme",
        r"dark mode",
        r"layout",
        r"navigation",
    ],
    "LOGIN": [
        r"log ?in",
        r"sign ?in",
        r"password",
        r"mpin",
        r"pin",
        r"login nahi",
        r"credential",
        r"username",
    ],
    "OTP": [r"\botp\b", r"one time password", r"otp nahi"],
    "UPI": [
        r"\bupi\b",
        r"\bvh\b",
        r"\bupa?y id\b",
        r"\bbhim\b",
        r"\bphone ?pe\b",
        r"\bg ?pay\b",
        r"\bpaytm\b",
        r"upi nahi",
        r"\bbhim upi\b",
    ],
    "NEFT": [r"\bneft\b"],
    "IMPS": [r"\bimps\b"],
    "RTGS": [r"\brtgs\b"],
    "ATM": [r"\batm\b", r"cash machine"],
    "CUSTOMER_SERVICE": [
        r"customer (?:care|service|support)",
        r"support team",
        r"helpline",
        r"no response",
        r"not responding",
        r"complaint",
        r"grievance",
        r"call (?:center|centre)",
    ],
    "TRANSACTION": [
        r"transaction",
        r"payment",
        r"pay",
        r"transfer",
        r"transaction fail\w*",
        r"payment fail\w*",
    ],
    "SECURITY": [
        r"security",
        r"secure",
        r"safe",
        r"breach",
        r"data leak",
        r"two factor",
        r"2fa",
    ],
    "FRAUD": [r"fraud\w*", r"scam\w*", r"cheat\w*", r"phishing", r"hacked?", r"chor"],
    "MONEY_TRANSFER": [
        r"money transfer",
        r"fund transfer",
        r"send money",
        r"remit",
        r"paisa bhej",
    ],
    "KYC": [
        r"\bkyc\b",
        r"re ?kyc",
        r"know your customer",
        r"aadhaar",
        r"\bpan card\b",
        r"video kyc",
    ],
    "ACCOUNT": [
        r"account",
        r"balance",
        r"statement",
        r"passbook",
        r"account open",
        r"saving account",
        r"current account",
    ],
    "LOAN": [r"\bloan\b", r"\bemi\b", r"interest rate", r"repay"],
    "CREDIT_CARD": [r"credit card", r"\bcc\b limit"],
    "NOTIFICATION": [r"notification", r"alert", r"push message", r"sms"],
    "SERVER": [r"server", r"maintenance", r"down time", r"downtime"],
    "CRASH": [r"crash\w*", r"force clos\w*", r"stopped working", r"not responding"],
    "FEATURE_REQUEST": [
        r"please add",
        r"feature request",
        r"should have",
        r"kindly add",
        r"update (?:the )?app",
    ],
    "CHARGES": [r"charge", r"fee", r"commission", r"charge bada", r"extra charge"],
}
ASPECT_RES = {a: compile_any(p) for a, p in ASPECT_PATTERNS.items()}

_NEG_CONTEXT = compile_any(
    [
        r"not work",
        r"n.?t work",
        r"issue",
        r"problem",
        r"fail",
        r"error",
        r"worst",
        r"bad",
        r"poor",
        r"slow",
        r"stuck",
        r"crash",
        r"nahi",
        r"useless",
        r"annoying",
        r"terrible",
        r"worst ever",
        r"can.?t",
        r"cannot",
        r"couldn.?t",
        r"didn.?t",
        r"no response",
        r"never",
    ]
)
_POS_CONTEXT = compile_any(
    [
        r"good",
        r"great",
        r"nice",
        r"easy",
        r"smooth",
        r"fast",
        r"love",
        r"best",
        r"excellent",
        r"awesome",
        r"helpful",
        r"perfect",
        r"useful",
        r"accha",
        r"badhiya",
        r"sahi",
    ]
)


def _window_sentiment(text: str, match: re.Match) -> str:
    """Local sentiment within a char window around the aspect mention."""
    lo = max(0, match.start() - 60)
    hi = min(len(text), match.end() + 60)
    ctx = text[lo:hi].lower()
    neg = len(_NEG_CONTEXT.findall(ctx))
    pos = len(_POS_CONTEXT.findall(ctx))
    if neg > pos:
        return "NEGATIVE"
    if pos > neg:
        return "POSITIVE"
    return "NEUTRAL"


def extract_aspects_single(text) -> dict:
    """Aspect extraction for one review (null-safe)."""
    if (
        text is None
        or (isinstance(text, float) and pd.isna(text))
        or not str(text).strip()
    ):
        return {"aspects": None, "aspect_count": 0, "aspect_sentiment": None}
    t = str(text)
    found: list[str] = []
    sentiments: dict[str, str] = {}
    for aspect, rex in ASPECT_RES.items():
        m = rex.search(t)
        if m:
            found.append(aspect)
            sentiments[aspect] = _window_sentiment(t, m)
    if not found:
        return {"aspects": None, "aspect_count": 0, "aspect_sentiment": None}
    return {
        "aspects": ",".join(found[:6]),
        "aspect_count": len(found),
        "aspect_sentiment": ";".join(f"{a}:{sentiments[a]}" for a in found[:6]),
    }


def detect_aspects_frame(
    df: pd.DataFrame, text_col: str = "review_text_model"
) -> pd.DataFrame:
    logger.info("Extracting aspects for %d rows ...", len(df))
    recs = df[text_col].map(extract_aspects_single)
    aspect_df = pd.DataFrame(recs.tolist(), index=df.index)
    out = pd.concat([df, aspect_df], axis=1)
    logger.info("Reviews with >=1 aspect: %d", int((out["aspect_count"] > 0).sum()))
    return out
