#!/usr/bin/env python3
"""Step 03b: Differential EDD restricted to SOLID-tumour lineages (BEACON MCMC).

Motivation: the original immune-hot stratification was confounded by
hematopoietic cancer cell lines (Lymphoid/Myeloid OncotreeLineage),
which transcriptionally resemble immune cells and inflate the ESTIMATE
immune score. The "immune-hot" group was ~31% hematopoietic vs.
~0.2% in the immune-cold group. As a result, the top differential EDDs
(IRF4, IKZF1, MYB, IKZF3) were lymphoid-identity essential genes —
real biology in lymphoma/myeloma cell lines, but not informative for
solid-tumour ICB response.

This script:
  1. Excludes Lymphoid and Myeloid lineages
  2. Recomputes ESTIMATE immune scores on solid-tumour expression
  3. Stratifies the remaining lines into immune-hot/cold
  4. Runs differential dependency correlation using BEACON Bayesian MCMC
     in each immune-context stratum, then compares posteriors via
     Fisher's z (method="beacon", consistent with the parent step 03).

Use --fast to fall back to Spearman+Fisher-z for development; the
default is full BEACON MCMC for publication.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, fdr_correction, get_logger
from data.depmap import load_cell_line_info, load_crispr, load_expression
from immune.beacon_immune import differential_edd
from immune.deconvolution import run_estimate, stratify_immune

log = get_logger("03b_solid")

HEME_LINEAGES = {"Lymphoid", "Myeloid"}


def spearman_per_group(expr, dep, ids_a, ids_b, genes):
    """Compute per-group Spearman rho and Fisher z differential test."""
    records = []
    for gene in genes:
        if gene not in expr.columns or gene not in dep.columns:
            continue
        ea = expr.loc[ids_a, gene].dropna()
        da = dep.loc[ids_a, gene].dropna()
        common_a = ea.index.intersection(da.index)
        eb = expr.loc[ids_b, gene].dropna()
        db = dep.loc[ids_b, gene].dropna()
        common_b = eb.index.intersection(db.index)
        if len(common_a) < 10 or len(common_b) < 10:
            continue
        ra, _ = stats.spearmanr(ea.loc[common_a], da.loc[common_a])
        rb, _ = stats.spearmanr(eb.loc[common_b], db.loc[common_b])
        if np.isnan(ra) or np.isnan(rb):
            continue
        # Fisher-z differential test
        za = np.arctanh(np.clip(ra, -0.999, 0.999))
        zb = np.arctanh(np.clip(rb, -0.999, 0.999))
        se = np.sqrt(1 / (len(common_a) - 3) + 1 / (len(common_b) - 3))
        z = (za - zb) / se
        p = 2 * stats.norm.sf(abs(z))
        records.append({
            "gene": gene,
            "rho_immune_hot": ra,
            "rho_immune_cold": rb,
            "delta_rho": ra - rb,
            "n_hot": len(common_a),
            "n_cold": len(common_b),
            "z_fisher": z,
            "pvalue": p,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["fdr"] = fdr_correction(df["pvalue"].values)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use Spearman+Fisher-z instead of BEACON MCMC")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    out = ensure_dir(Path(CFG["output_dir"]) / "immune_context_solid")

    log.info("Loading DepMap data")
    depmap_expr = load_expression()
    crispr = load_crispr()
    cell_info = load_cell_line_info()

    log.info("Excluding hematopoietic lineages: %s", HEME_LINEAGES)
    lineage_col = "OncotreeLineage" if "OncotreeLineage" in cell_info.columns else "PrimaryDisease"
    solid_ids = cell_info.index[~cell_info[lineage_col].isin(HEME_LINEAGES)]

    expr_solid = depmap_expr.loc[depmap_expr.index.intersection(solid_ids)]
    dep_solid = crispr.loc[crispr.index.intersection(solid_ids)]
    log.info("Solid-tumour cell lines: expr=%d, dep=%d",
             len(expr_solid), len(dep_solid))

    log.info("Re-running ESTIMATE on solid-tumour expression")
    solid_estimate = run_estimate(expr_solid)
    solid_estimate.to_csv(out / "solid_estimate_scores.csv")

    solid_status = stratify_immune(solid_estimate, method="median")
    solid_status.to_csv(out / "solid_immune_status.csv")
    n_hot = (solid_status == "immune_hot").sum()
    n_cold = (solid_status == "immune_cold").sum()
    log.info("Solid-tumour immune-hot: %d, immune-cold: %d", n_hot, n_cold)

    # Composition check
    hot_lineages = cell_info.loc[
        cell_info.index.intersection(solid_status[solid_status == "immune_hot"].index),
        lineage_col
    ].value_counts().head(10)
    log.info("Top 10 lineages in solid immune-hot:\n%s", hot_lineages.to_string())

    # Candidate genes
    beacon_dir = Path(CFG["output_dir"]) / "beacon_edd"
    fast_path = beacon_dir / "fast_screen_mrna.csv"
    if fast_path.exists():
        fast_df = pd.read_csv(fast_path)
        candidate_genes = fast_df[fast_df["rho"] < -0.15]["gene"].tolist()
        log.info("Candidate genes from fast screen: %d", len(candidate_genes))
    else:
        sig_path = beacon_dir / "beacon_edd_significant.csv"
        candidate_genes = pd.read_csv(sig_path)["gene"].unique().tolist()
        log.info("Candidate genes from significant EDD: %d", len(candidate_genes))

    common = expr_solid.index.intersection(dep_solid.index)
    ids_hot = common.intersection(solid_status[solid_status == "immune_hot"].index)
    ids_cold = common.intersection(solid_status[solid_status == "immune_cold"].index)
    log.info("Common solid-tumour lines: hot=%d cold=%d",
             len(ids_hot), len(ids_cold))

    if args.fast:
        log.info("Running Spearman+Fisher-z differential test (--fast)")
        diff = spearman_per_group(expr_solid, dep_solid, ids_hot, ids_cold,
                                  candidate_genes)
    else:
        log.info("Running BEACON Bayesian MCMC differential EDD (solid tumours)")
        diff = differential_edd(
            expr_solid, dep_solid, solid_status,
            group_a="immune_hot", group_b="immune_cold",
            method="beacon",
            candidate_genes=candidate_genes,
            n_jobs=args.n_jobs,
        )
    diff.to_csv(out / "differential_edd_solid_hot_vs_cold.csv", index=False)

    n_sig = (diff["fdr"] < 0.05).sum() if not diff.empty else 0
    log.info("Solid-tumour differential EDD: %d genes with FDR < 0.05", n_sig)

    if not diff.empty:
        log.info("Top 20 immune-hot specific (solid tumours):\n%s",
                 diff[diff["delta_rho"] < 0]
                 .sort_values("delta_rho")
                 .head(20)[["gene", "rho_immune_hot", "rho_immune_cold",
                            "delta_rho", "pvalue", "fdr"]]
                 .to_string(index=False))


if __name__ == "__main__":
    main()
