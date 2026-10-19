"""Page: Aspect & PageRank (plan s.49, s.18)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import ROOT, figure, get_filtered, load_table


def render() -> None:
    st.title("🧩 Aspect & PageRank")
    st.caption("PageRank measures STRUCTURAL CENTRALITY in the aspect "
               "co-occurrence network. It says nothing about sentiment "
               "(plan s.69).")
    df = get_filtered()

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews with >=1 aspect",
              f"{(df['aspect_count'] > 0).sum():,}",
              help="Denominator: all filtered reviews")
    c2.metric("Mean aspects / review",
              f"{df['aspect_count'].mean():.2f}")
    c3.metric("Distinct aspects seen",
              f"{df['aspects'].dropna().str.split(',').explode().nunique()}")

    st.subheader("Top aspects (frequency)")
    counts = (df["aspects"].dropna().str.split(",").explode()
              .value_counts().head(25))
    st.dataframe(counts.rename("mentions"), width="stretch")

    st.subheader("Global PageRank")
    pr = load_table("pagerank_global.csv")
    if pr.empty:
        st.warning("Run the full pipeline to generate PageRank tables.")
    else:
        st.dataframe(pr, width="stretch")
        st.caption("degree / weighted_degree = connectivity in the "
                   "co-occurrence graph; mention_count = raw mentions.")

    st.subheader("PageRank by bank")
    prb = load_table("pagerank_by_bank.csv")
    if not prb.empty:
        banks = sorted(prb["bank"].unique())
        pick = st.selectbox("Bank", banks)
        st.dataframe(prb[prb["bank"] == pick], width="stretch")

    st.subheader("Strongest aspect co-occurrences")
    edges_path = ROOT / "data" / "predictions" / "aspect_edges.parquet"
    if edges_path.exists():
        edges = pd.read_parquet(edges_path).head(25)
        st.dataframe(edges, width="stretch")
        st.caption("Undirected pairs (aspect_a < aspect_b) with "
                   "co-occurrence weight - the PageRank input graph.")
    else:
        st.info("`aspect_edges.parquet` not found - run "
                "`python -m src.pipeline`.")

    st.subheader("Aspect-level sentiment")
    rows = []
    for aspects, sents in zip(df["aspects"].dropna(),
                              df["aspect_sentiment"].dropna()):
        mapping = dict(p.split(":") for p in str(sents).split(";") if ":" in p)
        for a in str(aspects).split(","):
            if a in mapping:
                rows.append({"aspect": a, "aspect_sentiment": mapping[a]})
    if rows:
        adf = pd.DataFrame(rows)
        top = adf["aspect"].value_counts().head(15).index
        st.dataframe(
            adf[adf["aspect"].isin(top)]
            .groupby("aspect")["aspect_sentiment"].value_counts()
            .unstack(fill_value=0), width="stretch")
        st.caption("Aspect sentiment uses a local context window around each "
                   "mention (heuristic baseline).")

    st.divider()
    figure("12_pagerank_top_aspects.png")
    figure("10_aspect_frequency.png")
    figure("11_aspect_sentiment.png")


if __name__ == "__main__":
    render()
else:
    render()
