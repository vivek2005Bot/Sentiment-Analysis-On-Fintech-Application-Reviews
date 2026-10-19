"""Statistical analysis (plan section 34): Govt vs Private + effect sizes.

Framing: all comparisons are OBSERVATIONAL. Wording uses 'associated with' /
'observed difference'; no causal claims (plan Rule 10). Tests: chi-square for
categorical distributions, Mann-Whitney U for ratings. Multiple comparisons
corrected with Benjamini-Hochberg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from src.utils import setup_logging

logger = setup_logging("statistical_analysis")


def benjamini_hochberg(pvals: list[float] | pd.Series) -> np.ndarray:
    """BH-corrected q-values; NaN inputs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = len(pm)
    if n == 0:
        return out
    order = np.argsort(pm)
    ranked = pm[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    out[mask] = q
    return out


def _cramers_v(cont: np.ndarray) -> float:
    n = cont.sum()
    r, c = cont.shape
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return 0.0
    chi2 = chi2_contingency(cont)[0]
    return float(np.sqrt(chi2 / denom))


def govt_vs_private_table(master: pd.DataFrame) -> pd.DataFrame:
    """Table 5 (plan s.55): metric-wise group comparison with effect size."""
    rows = []

    def add(metric, g_val, p_val, diff, test, stat, p, effect, n_g, n_p):
        rows.append(
            {
                "metric": metric,
                "government": g_val,
                "private": p_val,
                "difference": diff,
                "statistical_test": test,
                "test_statistic": stat,
                "p_value": p,
                "effect_size": effect,
                "n_government": n_g,
                "n_private": n_p,
            }
        )

    g = master[master["bank_type"] == "GOVERNMENT"]
    p = master[master["bank_type"] == "PRIVATE"]

    # Negative rate (chi-square + Cramer's V)
    def neg_rate(d):
        s = d["sentiment"]
        return float((s == "NEGATIVE").mean()) if len(d) else np.nan

    add(
        "negative_rate",
        round(neg_rate(g), 4),
        round(neg_rate(p), 4),
        round(neg_rate(g) - neg_rate(p), 4),
        "chi_square",
        None,
        None,
        None,
        len(g),
        len(p),
    )
    cont = np.array(
        [
            [
                (g["sentiment"] == "NEGATIVE").sum(),
                (g["sentiment"] != "NEGATIVE").sum(),
            ],
            [
                (p["sentiment"] == "NEGATIVE").sum(),
                (p["sentiment"] != "NEGATIVE").sum(),
            ],
        ]
    )
    chi2, pval, _, _ = chi2_contingency(cont)
    rows[-1].update(
        test_statistic=float(chi2),
        p_value=float(pval),
        effect_size=round(_cramers_v(cont), 4),
    )

    # Failure rate
    for metric, col, yes in [
        ("failure_rate", "failure_flag", "YES"),
        ("critical_failure_rate", "critical_failure", True),
    ]:
        gr = float((g[col] == yes).mean()) if len(g) else np.nan
        pr_ = float((p[col] == yes).mean()) if len(p) else np.nan
        add(
            metric,
            round(gr, 4),
            round(pr_, 4),
            round(gr - pr_, 4),
            "chi_square",
            None,
            None,
            None,
            len(g),
            len(p),
        )
        cont = np.array(
            [
                [(g[col] == yes).sum(), (g[col] != yes).sum()],
                [(p[col] == yes).sum(), (p[col] != yes).sum()],
            ]
        )
        chi2, pval, _, _ = chi2_contingency(cont)
        rows[-1].update(
            test_statistic=float(chi2),
            p_value=float(pval),
            effect_size=round(_cramers_v(cont), 4),
        )

    # Mean rating (Mann-Whitney U + rank-biserial correlation)
    gr = g["rating"].dropna()
    pr_ = p["rating"].dropna()
    add(
        "mean_rating",
        round(float(gr.mean()), 3),
        round(float(pr_.mean()), 3),
        round(float(gr.mean() - pr_.mean()), 3),
        "mannwhitney_u",
        None,
        None,
        None,
        len(gr),
        len(pr_),
    )
    if len(gr) > 20 and len(pr_) > 20:
        stat, pval = mannwhitneyu(gr, pr_, alternative="two-sided")
        rbc = 1 - 2 * stat / (len(gr) * len(pr_))
        rows[-1].update(
            test_statistic=float(stat),
            p_value=float(pval),
            effect_size=round(float(rbc), 4),
        )

    # Risk mix among failure rows
    fg = g[g["failure_flag"] == "YES"]
    fp = p[p["failure_flag"] == "YES"]
    if len(fg) and len(fp):
        gt = float((fg["primary_risk"] == "TECHNICAL_PERFORMANCE_RISK").mean())
        pt = float((fp["primary_risk"] == "TECHNICAL_PERFORMANCE_RISK").mean())
        add(
            "technical_share_among_failures",
            round(gt, 4),
            round(pt, 4),
            round(gt - pt, 4),
            "chi_square",
            None,
            None,
            None,
            len(fg),
            len(fp),
        )
        cont = np.array(
            [
                [
                    (fg["primary_risk"] == "TECHNICAL_PERFORMANCE_RISK").sum(),
                    (fg["primary_risk"] != "TECHNICAL_PERFORMANCE_RISK").sum(),
                ],
                [
                    (fp["primary_risk"] == "TECHNICAL_PERFORMANCE_RISK").sum(),
                    (fp["primary_risk"] != "TECHNICAL_PERFORMANCE_RISK").sum(),
                ],
            ]
        )
        chi2, pval, _, _ = chi2_contingency(cont)
        rows[-1].update(
            test_statistic=float(chi2),
            p_value=float(pval),
            effect_size=round(_cramers_v(cont), 4),
        )

    table = pd.DataFrame(rows)
    q = benjamini_hochberg(table["p_value"].fillna(np.nan))
    table["q_value_bh"] = np.round(q, 5)
    return table
