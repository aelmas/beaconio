#!/usr/bin/env python3
"""Step 09: Real-tumor-microenvironment corroboration of the lineage-corrected
solid-tumor immune-hot-specific hits (addresses referee concerns C14/C5).

Cell lines have no microenvironment, so the cell-line immune score is a
tumor-cell-intrinsic proxy that co-varies with lineage. Here we ask whether
the 13 lineage-corrected hot-specific genes are associated with immune context
in REAL TCGA tumors (which do have a TME), *after adjusting for cancer type
(lineage)*. The prediction that distinguishes a genuine immune-context gene
from a lineage artifact:

  - lineage-driven hits (IRF4, MITF, SOX10; melanocyte program) should LOSE
    their immune association once cancer type is included as a covariate;
  - lineage-independent hits (TEAD3, USP18, BHLHE41, CDH1, ...) should RETAIN a
    significant association with immune / cytolytic activity across cancer types.

For each gene we regress its expression on the tumor immune score (and on the
Rooney cytolytic-activity score, CYT) with and without a cancer-type covariate,
using pooled TCGA tumors across the analysis cancer types. A background of
random genes calibrates how often a gene is "immune-associated after lineage
adjustment" by chance.

Output:
  - analysis/out/clinical/tcga_tme_corroboration.csv
  - analysis/out/figures/fig8_tcga_tme_corroboration.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, fdr_correction, get_logger
from data.tcga import load_tcga_clinical, load_tcga_expression
from immune.deconvolution import run_estimate

log = get_logger("09_tcga_tme")

MELANOCYTE_LINEAGE = {"IRF4", "MITF", "SOX10"}  # expected lineage-driven (C5)
CYT_GENES = ["GZMA", "PRF1"]
MIN_PER_TYPE = 30


def _zscore_within(df, value_col, group_col):
    """Z-score a column within each group (cancer type) to remove lineage means."""
    g = df.groupby(group_col)[value_col]
    return (df[value_col] - g.transform("mean")) / (g.transform("std") + 1e-9)


def _fit(df, gene_col, predictor):
    """Return (beta_unadj, p_unadj, beta_adj, p_adj) for gene ~ predictor,
    without and with C(cancer_type)."""
    sub = df[[gene_col, predictor, "cancer_type"]].dropna()
    if sub[gene_col].std() < 1e-8 or sub[predictor].std() < 1e-8 or len(sub) < 50:
        return (np.nan, np.nan, np.nan, np.nan)
    sub = sub.rename(columns={gene_col: "y", predictor: "x"})
    try:
        m1 = smf.ols("y ~ x", data=sub).fit()
        m2 = smf.ols("y ~ x + C(cancer_type)", data=sub).fit()
        return (m1.params["x"], m1.pvalues["x"], m2.params["x"], m2.pvalues["x"])
    except Exception as exc:  # noqa: BLE001
        log.warning("fit failed for %s: %s", gene_col, exc)
        return (np.nan, np.nan, np.nan, np.nan)


def main():
    out = ensure_dir(Path(CFG["output_dir"]) / "clinical")
    fig_dir = ensure_dir(Path(CFG["output_dir"]) / "figures")

    # ── 13 lineage-corrected hot-specific hits ──────────────────────
    pdiff = pd.read_csv(Path(CFG["output_dir"]) / "immune_context_solid"
                        / "posterior_difference_solid.csv")
    hits = pdiff[pdiff["posterior_significant"]]["gene"].tolist()
    log.info("Hot-specific hits to corroborate: %s", hits)

    # ── Load full TCGA expression once + cancer-type labels ─────────
    log.info("Loading full TCGA expression (one pass)")
    expr = load_tcga_expression()                      # samples x genes (symbols)
    clin = load_tcga_clinical(Path(CFG["data_dir"]))
    type_col = next((c for c in ("cancer_type", "_primary_disease", "project_id",
                                 "disease") if c in clin.columns), None)
    if type_col is None:
        log.error("No cancer-type column in clinical; columns=%s", list(clin.columns)[:20])
        return
    common = expr.index.intersection(clin.index)
    expr = expr.loc[common]
    cancer = clin.loc[common, type_col].astype(str)
    # Keep cancer types with enough tumors
    vc = cancer.value_counts()
    keep_types = vc[vc >= MIN_PER_TYPE].index
    mask = cancer.isin(keep_types)
    expr = expr.loc[mask]
    cancer = cancer.loc[mask]
    log.info("TCGA tumors: %d across %d cancer types", len(expr), cancer.nunique())

    # ── Per-tumor immune score and cytolytic activity ───────────────
    immune = run_estimate(expr)
    immune_col = "ImmuneScore" if "ImmuneScore" in immune.columns else immune.columns[0]
    cyt_av = [g for g in CYT_GENES if g in expr.columns]
    cyt = np.log1p(expr[cyt_av]).mean(axis=1)

    base = pd.DataFrame({
        "cancer_type": cancer,
        "immune": immune[immune_col].reindex(expr.index),
        "CYT": cyt.reindex(expr.index),
    })
    # z-score predictors within cancer type (remove lineage mean differences)
    base["immune_z"] = _zscore_within(base, "immune", "cancer_type")
    base["CYT_z"] = _zscore_within(base, "CYT", "cancer_type")

    def evaluate(genes, label):
        rows = []
        for g in genes:
            if g not in expr.columns:
                continue
            df = base.copy()
            df[g] = _zscore_within(pd.DataFrame({g: expr[g], "cancer_type": cancer}), g, "cancer_type")
            bi_u, pi_u, bi_a, pi_a = _fit(df, g, "immune_z")
            bc_u, pc_u, bc_a, pc_a = _fit(df, g, "CYT_z")
            rows.append({
                "gene": g, "set": label,
                "immune_beta_unadj": bi_u, "immune_p_unadj": pi_u,
                "immune_beta_adj": bi_a, "immune_p_adj": pi_a,
                "CYT_beta_adj": bc_a, "CYT_p_adj": pc_a,
                "melanocyte_lineage": g in MELANOCYTE_LINEAGE,
            })
        return pd.DataFrame(rows)

    res = evaluate(hits, "hot_specific")

    # ── Background distribution of lineage-adjusted effect sizes ─────
    rng = np.random.default_rng(42)
    pool = [g for g in expr.columns if g not in hits]
    bg_genes = list(rng.choice(pool, size=min(300, len(pool)), replace=False))
    bg = evaluate(bg_genes, "background")
    bg.to_csv(out / "tcga_tme_background.csv", index=False)

    # At n~10.5k, p-values are uninformative (almost everything is
    # "significant"); the meaningful quantity is the EFFECT SIZE relative to
    # the random-gene background. We therefore compare each hit's
    # lineage-adjusted standardized beta against the background |beta|
    # distribution and flag genes whose |beta| exceeds its 95th percentile.
    bg_abs = bg["immune_beta_adj"].abs().dropna()
    thr95 = float(np.percentile(bg_abs, 95))
    bg_frac_sig = float((fdr_correction(bg["immune_p_adj"].fillna(1).values) < 0.05).mean())
    res["bg_abs_percentile"] = res["immune_beta_adj"].abs().apply(
        lambda b: float((bg_abs < abs(b)).mean()) if pd.notna(b) else np.nan)
    res["immune_associated_effectsize"] = res["immune_beta_adj"].abs() > thr95
    res["immune_fdr_adj"] = fdr_correction(res["immune_p_adj"].fillna(1).values)
    res = res.sort_values("immune_beta_adj", ascending=False)
    res.to_csv(out / "tcga_tme_corroboration.csv", index=False)

    # ── Summary ─────────────────────────────────────────────────────
    log.info("=== Real-TME (TCGA) lineage-adjusted immune effect sizes ===")
    log.info("Background |beta| 95th percentile (effect-size threshold): %.3f", thr95)
    log.info("NOTE: at n=%d, %.0f%% of random genes reach FDR<0.05 -> p-values uninformative; "
             "we use effect size vs background.", len(expr), 100 * bg_frac_sig)
    tumor_intrinsic = res[~res["immune_associated_effectsize"]]["gene"].tolist()
    immune_assoc = res[res["immune_associated_effectsize"]]["gene"].tolist()
    log.info("Hits with LARGE immune effect (track infiltrate; >95th pct bg): %s", immune_assoc)
    log.info("Hits that are TUMOR-INTRINSIC (immune effect within background): %s", tumor_intrinsic)
    log.info("\n%s", res[["gene", "melanocyte_lineage", "immune_beta_adj",
                          "bg_abs_percentile", "immune_associated_effectsize",
                          "CYT_beta_adj"]].round(4).to_string(index=False))

    # ── Figure 8: effect size vs background ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    # background distribution as a vertical strip at x=0
    yb = bg["immune_beta_adj"].dropna().values
    ax.scatter(np.random.default_rng(1).normal(0, 0.04, len(yb)), yb,
               s=8, c="#bbbbbb", alpha=0.5, label="random genes (background)")
    ax.axhline(thr95, ls=":", c="grey", lw=0.8)
    ax.axhline(-thr95, ls=":", c="grey", lw=0.8)
    ax.axhline(0, ls="-", c="grey", lw=0.5)
    for i, (_, r) in enumerate(res.iterrows()):
        if pd.isna(r["immune_beta_adj"]):
            continue
        c = "#d73027" if r["melanocyte_lineage"] else (
            "#1a9850" if not r["immune_associated_effectsize"] else "#4575b4")
        xj = 1 + (i % 5) * 0.0
        ax.scatter(1, r["immune_beta_adj"], s=60, c=c, edgecolors="white", zorder=3)
        ax.annotate(r["gene"], (1.04, r["immune_beta_adj"]), fontsize=7, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Background\n(300 random genes)", "Hot-specific hits"])
    ax.set_ylabel("Lineage-adjusted standardized immune-association $\\beta$ (TCGA)")
    ax.set_xlim(-0.3, 1.4)
    ax.set_title("Real-TME corroboration (TCGA, n=10{,}534 tumors)\n"
                 "Effect size of immune association after lineage adjustment\n"
                 "red=melanocyte-lineage; green=tumor-intrinsic (within background); blue=immune-associated",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig8_tcga_tme_corroboration.pdf")
    fig.savefig(fig_dir / "fig8_tcga_tme_corroboration.png", dpi=150)
    plt.close(fig)
    log.info("Saved fig8_tcga_tme_corroboration")


if __name__ == "__main__":
    main()
