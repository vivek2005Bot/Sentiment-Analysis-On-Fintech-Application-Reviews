"""Phase 5: language, script, Hinglish and transliteration detection (s.9).

Fast, deterministic, corpus-scale heuristics:
- Unicode-script shares classify native Indic-script reviews (Hindi, Tamil...);
- romanized-Hindi / romanized-Tamil lexicon ratios classify Hinglish and
  Tanglish in Latin script (Latin script NEVER implies English - Rule s.9);
- langdetect refinement is optional/off by default (config, sampled).
Every output field keeps null-safe behaviour on empty text.
"""

from __future__ import annotations

import re

import pandas as pd

import src.config as sc
from src.utils import setup_logging

logger = setup_logging("language_detector")

SCRIPT_RANGES: dict[str, str] = {
    "Devanagari": "\u0900-\u097f",
    "Bengali": "\u0980-\u09ff",
    "Gurmukhi": "\u0a00-\u0a7f",
    "Gujarati": "\u0a80-\u0aff",
    "Tamil": "\u0b80-\u0bff",
    "Telugu": "\u0c00-\u0c7f",
    "Kannada": "\u0c80-\u0cff",
    "Malayalam": "\u0d00-\u0d7f",
    "Oriya": "\u0b00-\u0b7f",
}
SCRIPT_TO_LANG = {
    "Devanagari": "Hindi",
    "Bengali": "Bengali",
    "Gurmukhi": "Punjabi",
    "Gujarati": "Gujarati",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Oriya": "Oriya",
}

# Romanized Hindi tokens (transliterated Hindustani common in reviews).
_HINDI_WORDS = """
nahi nahiin nhi hai hain ho hogi hoga hona hu hun hoon tha thi the
kar karo karna karne krna krdo kiya kiye karti karta karte
mat mana mangta bahut bahot bhut bohot accha achha acha acchi
theek thik sahi galat galti bekaar bakwas faltu ghatiya
paisa paise paisay rupaye rupya chahiye chahiya jaldi abhi
kyu kyun kyon kaise kaisa kaisi kaha kahan kab na naa haan han
mere mera meri hum ap aap tum ye yeh wo woh is iss us iska uska isko usko
ka ki ke ko se par bhi hi to tho batao bata dikha dekho
samajh samjho suno lijiye dijiye kijiye banaya band chalu
khol kholo khulta khul bandh dhang tarika tareeka
katu katt gaya gayi hua hui raha rahi lagta lag raha
dhoka dhokha cheat chor chori fraud hack hackli
nahi ho raha pa open khul login otp badal badlo banao dikhao chal ya
""".split()
HINDI_LEXICON = frozenset(_HINDI_WORDS)

_TAMIL_WORDS = """
romba irukku iruku irukka irukum venam venum vendam pannu pannunga
pannirukku panne pannitu aana ana illa illama illathi seri sari podu
podunga kudu kudunga vaanga thanga vaa thaa theriyum theriyala puriyala
puriyathu yenna enna eppadi epdi sollu solru sollunga vanakkam machi
machan anna thambi akka thangachi aama ama illaye po poga mudiyala
kadai pasi venumilla veendum kitti
""".split()
TAMIL_LEXICON = frozenset(_TAMIL_WORDS)

_LATIN_TOKEN_RE = re.compile(r"[A-Za-z']+")
_ENGLISH_STOPWORDS = frozenset("""
the a an and or of to in is are was were be been it this that with for on at
from by as but not no yes you i we they he she my your our their me us him
her them what which who whom how when where why all any both each few more
most other some such only own same so than too very can will just should now
app please bank account money payment good use using update update
""".split())

_hindi_re = re.compile(
    r"\b(?:" + "|".join(sorted(HINDI_LEXICON)) + r")\b", re.IGNORECASE
)
_tamil_re = re.compile(
    r"\b(?:" + "|".join(sorted(TAMIL_LEXICON)) + r")\b", re.IGNORECASE
)
_script_res = {name: re.compile(f"[{rng}]") for name, rng in SCRIPT_RANGES.items()}
_letter_re = re.compile(r"[^\W\d_]", re.UNICODE)


def detect_single(text) -> dict:
    """Return language fields for one text (null-safe)."""
    cfg = sc.get_config()["language"]
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return {
            "language": None,
            "script": None,
            "is_code_mixed": None,
            "is_transliterated": None,
            "language_confidence": None,
            "hinglish_ratio": None,
        }
    t = str(text)
    n_letters = len(_letter_re.findall(t))
    shares = {
        name: (len(res.findall(t)) / n_letters) if n_letters else 0.0
        for name, res in _script_res.items()
    }
    top_script, top_share = max(shares.items(), key=lambda kv: kv[1])

    tokens = [w.lower() for w in _LATIN_TOKEN_RE.findall(t)]
    n_tokens = len(tokens)

    # Native Indic script can be classified even with zero Latin tokens
    if n_letters and top_share >= cfg["script_majority_threshold"]:
        lang = SCRIPT_TO_LANG[top_script]
        latin_share = sum(1 for c in t if c.isascii() and c.isalpha()) / max(
            n_letters, 1
        )
        code_mixed = latin_share >= 0.05
        return {
            "language": lang,
            "script": top_script,
            "is_code_mixed": code_mixed,
            "is_transliterated": False,
            "language_confidence": round(min(0.95, 0.6 + top_share), 3),
            "hinglish_ratio": 0.0,
        }

    if n_tokens < cfg["min_tokens_for_lexicon"]:
        return {
            "language": "Unknown",
            "script": "Unknown",
            "is_code_mixed": False,
            "is_transliterated": False,
            "language_confidence": 0.0,
            "hinglish_ratio": 0.0,
        }

    latin_share = sum(1 for c in t if c.isascii() and c.isalpha()) / max(n_letters, 1)

    hindi_ratio = len(_hindi_re.findall(t.lower())) / max(n_tokens, 1)
    tamil_ratio = len(_tamil_re.findall(t.lower())) / max(n_tokens, 1)

    base = {"hinglish_ratio": round(hindi_ratio, 4)}
    if top_share >= cfg["script_majority_threshold"]:
        lang = SCRIPT_TO_LANG[top_script]
        code_mixed = latin_share >= 0.05
        base.update(
            language=lang,
            script=top_script,
            is_code_mixed=code_mixed,
            is_transliterated=False,
            language_confidence=round(min(0.95, 0.6 + top_share), 3),
        )
        return base

    # Latin-dominant: romanized Indic lexicons decide (never assume English)
    english_present = any(w in _ENGLISH_STOPWORDS for w in tokens)
    if (
        hindi_ratio >= cfg["tanglish_ratio_threshold"]
        and tamil_ratio >= cfg["tanglish_ratio_threshold"]
    ):
        lang, script, translit, code_mixed = "Other_Indian", "Latin", True, True
        conf = min(0.9, 0.55 + hindi_ratio + tamil_ratio)
    elif hindi_ratio >= cfg["hinglish_ratio_threshold"]:
        lang, script = "Hinglish", "Latin"
        translit = hindi_ratio >= 0.20
        code_mixed = english_present
        conf = min(0.95, 0.55 + hindi_ratio)
    elif tamil_ratio >= cfg["tanglish_ratio_threshold"]:
        lang, script, translit, code_mixed = "Tanglish", "Latin", True, english_present
        conf = min(0.95, 0.55 + tamil_ratio)
    else:
        lang, script, translit, code_mixed = "English", "Latin", False, False
        conf = min(0.95, 0.5 + 0.45 * (1.0 - hindi_ratio - tamil_ratio))
    base.update(
        language=lang,
        script=script,
        is_code_mixed=code_mixed,
        is_transliterated=bool(translit),
        language_confidence=round(conf, 3),
    )
    return base


def detect_language_frame(
    df: pd.DataFrame, text_col: str = "review_text_model"
) -> pd.DataFrame:
    """Add language/script/code-mixed fields to a dataframe copy."""
    logger.info("Detecting language for %d rows ...", len(df))
    fields = df[text_col].map(detect_single)
    lang_df = pd.DataFrame(fields.tolist(), index=df.index)
    out = pd.concat([df, lang_df], axis=1)
    logger.info(
        "Language distribution: %s",
        out["language"].value_counts(dropna=False).to_dict(),
    )
    return out
