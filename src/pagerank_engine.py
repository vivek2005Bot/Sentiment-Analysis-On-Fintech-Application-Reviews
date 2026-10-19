"""PageRank engine (plan section 18/49): aspect co-occurrence network.

Builds an undirected weighted graph where nodes are aspects and edges
connect aspects co-occurring in the same review (edge weight = co-occurrence
count). Runs networkx PageRank globally and per bank group
(Government vs Private) + per bank. Structural centrality only - PageRank
says nothing about sentiment (plan section 69).
"""

from __future__ import annotations


import networkx as nx
import pandas as pd

import src.config as sc
from src.utils import setup_logging

logger = setup_logging("pagerank_engine")


def build_cooccurrence(sub: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for aspects in sub["aspects"].dropna():
        aspects = [a for a in str(aspects).split(",") if a]
        for a in aspects:
            if not G.has_node(a):
                G.add_node(a, count=0)
            G.nodes[a]["count"] += 1
        for i, a in enumerate(aspects):
            for b in aspects[i + 1:]:
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
    return G


def _graph_table(G: nx.Graph, pr: dict) -> pd.DataFrame:
    rows = []
    for node in G.nodes:
        rows.append(
            {
                "aspect": node,
                "pagerank": round(float(pr.get(node, 0.0)), 6),
                "degree": int(G.degree(node)),
                "weighted_degree": int(round(float(G.degree(node, weight="weight")))),
                "mention_count": int(G.nodes[node].get("count", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("pagerank", ascending=False)


def compute_pagerank(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Global + per-group + per-bank PageRank tables (writes CSVs)."""
    paths = sc.get_paths()
    damping = float(sc.get_config()["pagerank"]["damping"])
    tables: dict[str, pd.DataFrame] = {}

    def run(sub: pd.DataFrame, name: str, fname: str) -> None:
        if sub.empty or sub["aspects"].notna().sum() < 5:
            logger.warning("PageRank skipped for %s (too few aspects)", name)
            return
        G = build_cooccurrence(sub)
        if len(G) == 0:
            return
        try:
            pr = nx.pagerank(G, alpha=damping, weight="weight")
        except Exception:
            pr = {node: 1.0 / len(G) for node in G.nodes}
        t = _graph_table(G, pr)
        t.to_csv(paths["tables_dir"] / fname, index=False)
        tables[name] = t

    run(master, "global", "pagerank_global.csv")
    for group, sub in master.groupby("bank_type", observed=True):
        run(sub, f"group_{group}", f"pagerank_group_{group}.csv")
    for bank, sub in master.groupby("bank", observed=True):
        safe = str(bank).replace(" ", "_")
        run(sub, f"bank_{bank}", f"pagerank_bank_{safe}.csv")

    logger.info("PageRank computed for %d views", len(tables))
    return tables


def top_edges(sub: pd.DataFrame, k: int = 25) -> pd.DataFrame:
    G = build_cooccurrence(sub)
    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    df = pd.DataFrame(edges, columns=["aspect_a", "aspect_b", "weight"])
    return df.sort_values("weight", ascending=False).head(k)
