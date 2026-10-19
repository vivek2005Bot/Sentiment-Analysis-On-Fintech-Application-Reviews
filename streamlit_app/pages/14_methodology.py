"""Page: Data & Methodology (plan s.49, s.58, s.69)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from components.common import ROOT, get_filtered


def render() -> None:
    st.title("📚 Data & Methodology")
    df = get_filtered()

    st.subheader("Data sources")
    inv_path = ROOT / "data" / "inventory" / "data_inventory.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        usable = inv[inv["usable"] == True]  # noqa: E712
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw files discovered", len(inv))
        c2.metric("Usable files", len(usable))
        c3.metric("Files reported unusable", len(inv) - len(usable),
                  help="Empty/BOM-only or unreadable - reported, never "
                       "fabricated (plan Rule 6)")
        st.dataframe(inv[["source_file", "bank", "bank_type", "rows",
                          "rows_source", "usable", "status"]],
                     width="stretch")
    else:
        st.warning("`data/inventory/data_inventory.csv` not found.")

    st.subheader("Corpus in view")
    m = st.columns(4)
    m[0].metric("Reviews", f"{len(df):,}")
    m[1].metric("Banks", df["bank"].nunique())
    d = pd.to_datetime(df["review_date"], errors="coerce")
    m[2].metric("Earliest", str(d.min())[:10] if d.notna().any() else "—")
    m[3].metric("Latest", str(d.max())[:10] if d.notna().any() else "—")

    st.subheader("Banks")
    st.dataframe(df.groupby(["bank", "bank_type"]).size().rename("reviews"),
                 width="stretch")

    st.subheader("Label definitions (plan s.69 - kept separate)")
    st.markdown("""
| Concept | Definition |
|---|---|
| **Sentiment** | positive / neutral / negative attitude |
| **Operational friction** | inconvenience, UX issue, feature request, non-critical dissatisfaction |
| **Failure** | significant system/transaction problem (crash, server down, failed payment, unauthorised debit, OTP/login failure) |
| **Critical failure** | transaction failure or unauthorised/money-loss signal |
| **Technical Performance Risk** | system/app/server/performance failure (Tier 2) |
| **Financial Security Risk** | fraud, unauthorised transaction, money lost/stuck, security/data-breach concern (Tier 2) |
| **Severity** | project annotation policy: 1 suggestion … 5 financial/security incident |
| **PageRank** | structural centrality in the aspect co-occurrence network (says nothing about sentiment) |

Primary risk is mutually exclusive (financial wins when both fire) so the
official two-class benchmark has no overlap; `secondary_risk` and
`risk_overlap` retain the ambiguity (plan s.14).
""")

    st.subheader("Model names / versions in use")
    src = (df["sentiment_model_source"].dropna().value_counts().rename("rows")
           .to_frame())
    src["component"] = "sentiment"
    for col, comp in [("failure_model_source", "failure"),
                      ("risk_model_source", "risk"),
                      ("emotion_model_source", "emotion")]:
        if col in df.columns:
            part = df[col].dropna().value_counts().rename("rows").to_frame()
            part["component"] = comp
            src = pd.concat([src, part])
    st.dataframe(src.reset_index(names="model_source")[
        ["component", "model_source", "rows"]], width="stretch")
    st.caption("Deterministic `rules_*` sources are bootstrap baselines; "
               "`transformer` sources indicate a fine-tuned model produced "
               "that field (plan Rule 8).")

    st.subheader("Data quality report")
    qp = ROOT / "data" / "processed" / "quality_report.json"
    if qp.exists():
        with open(qp, encoding="utf-8") as fh:
            rep = json.load(fh)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total rows", f"{rep['total_rows']:,}")
        q2.metric("Empty text", f"{rep['empty_text_rows']:,}")
        q3.metric("Duplicate review IDs", f"{rep['duplicate_review_ids_rows']:,}")
        q4.metric("Duplicate normalised text",
                  f"{rep['duplicate_normalized_text_rows']:,}")
        st.caption("Duplicate text is COUNTED, never silently dropped "
                   "(plan s.7); near-identical reviews are not automatically "
                   "duplicates.")
        with st.expander("Artifact counts (URLs, emails, phones, emojis, HTML)"):
            st.json(rep.get("artifact_counts", {}))
        with st.expander("Missingness"):
            st.json(rep.get("missingness", {}))
    else:
        st.warning("`quality_report.json` not found - run "
                   "`python -m src.data_validator`.")

    st.subheader("Evaluation methodology")
    st.markdown("""
- Held-out test split only; no claim of superiority without it (Rule 9).
- Metrics: accuracy, macro/weighted F1, macro precision/recall, and per-class
  recall for the imbalanced risk task (s.22).
- Split designs: random, time-based, and cross-bank (s.26).
- Language-specific subsets reported separately (s.25).
- Flat vs hierarchical compared on the same held-out rows (s.24).
- Multi-comparison p-values corrected with Benjamini-Hochberg (s.34).
- The risk-sensitive cost matrix is a documented sensitivity analysis, not an
  objective cost (s.23).
""")

    st.subheader("Removal audit trail")
    rl = ROOT / "data" / "processed" / "removal_log.csv"
    if rl.exists():
        log = pd.read_csv(rl)
        if log.empty:
            st.info("No rows removed: every raw record is present in the "
                    "master dataset (plan Rule 5).")
        else:
            st.dataframe(log, width="stretch")
    st.caption("Any future removal is appended here with "
               "before/removed/after counts and a reason.")

    st.subheader("Known limitations")
    st.markdown("""
- **Rule-based baselines are heuristic.** Sentiment/failure/risk/emotion use
  versioned lexicons (see the model table above) and are bootstrap signals,
  not ground truth. Transformer artefacts replace them once trained.
- **Silver labels are not human annotations.** The annotation sample ships
  with blank human label columns; `annotation_sample_silver.csv` is marked
  `label_source=silver_rules_v1`, so any benchmark on it is provisional.
- **Six private-bank files were empty at ingestion** (HDFC, Kotak, IDFC
  First, Federal, Bandhan, City Union). They are reported unusable rather
  than fabricated, so the private side of the corpus is 3 banks only.
- **Observational data only.** No causal claims; response-latency findings
  are associations.
- **No follow-up snapshots**, so rating/sentiment *change* after a developer
  reply cannot be computed from this corpus (only levels).
- **Language detection is heuristic** (lexicon + Unicode script), not a
  calibrated language-ID model; very short reviews are `Unknown`.
- **PageRank** describes aspect co-occurrence structure, not importance to
  users or sentiment.
""")

if __name__ == "__main__":
    render()
else:
    render()
