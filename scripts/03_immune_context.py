#!/usr/bin/env python3
"""Step 03: Immune contextualization of BEACON EDD targets.

  A. TME deconvolution of TCGA tumours and DepMap cell lines
  B. Immune evasion programme scoring
  C. Stratify into immune-hot / immune-cold
  D. Differential EDD (hot vs cold)
  E. Evasion-correlated EDD
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from data.depmap import load_cell_line_info, load_crispr, load_expression
from data.tcga import load_tcga_expression
from immune.beacon_immune import differential_edd, evasion_correlated_edd
from immune.deconvolution import (
    run_estimate,
    run_mcpcounter,
    score_immune_evasion,
    stratify_immune,
)

log = get_logger("03_immune")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=CFG["data_dir"])
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    args = parser.parse_args()

    out = ensure_dir(Path(args.output_dir) / "immune_context")

    # ── A. TME deconvolution ───────────────────────────────────────────
    log.info("=== TME deconvolution (TCGA) ===")
    tcga_expr = load_tcga_expression(cancer_types=CFG["tcga"]["cancer_types"])

    estimate_scores = run_estimate(tcga_expr)
    estimate_scores.to_csv(out / "tcga_estimate_scores.csv")

    try:
        mcpcounter_scores = run_mcpcounter(tcga_expr)
        mcpcounter_scores.to_csv(out / "tcga_mcpcounter_scores.csv")
    except Exception as exc:
        log.warning("MCPcounter failed (%s), continuing with ESTIMATE only", exc)

    # ── B. Immune evasion scoring ──────────────────────────────────────
    log.info("=== Immune evasion scoring ===")
    evasion_tcga = score_immune_evasion(tcga_expr)
    evasion_tcga.to_csv(out / "tcga_evasion_scores.csv")

    # Also score DepMap cell lines (for EDD correlation)
    depmap_expr = load_expression()
    evasion_depmap = score_immune_evasion(depmap_expr)
    evasion_depmap.to_csv(out / "depmap_evasion_scores.csv")

    # ── C. Stratification ──────────────────────────────────────────────
    log.info("=== Immune stratification ===")
    immune_status = stratify_immune(estimate_scores, method="median")
    immune_status.to_csv(out / "tcga_immune_status.csv")
    log.info("Immune-hot: %d, Immune-cold: %d",
             (immune_status == "immune_hot").sum(),
             (immune_status == "immune_cold").sum())

    # ── D. Differential EDD (hot vs cold in DepMap cell lines) ─────────
    log.info("=== Differential EDD (BEACON MCMC) ===")
    crispr = load_crispr()
    cell_info = load_cell_line_info()

    # Use ESTIMATE on DepMap expression for stratification
    depmap_estimate = run_estimate(depmap_expr)
    depmap_immune_status = stratify_immune(depmap_estimate, method="median")

    # Load pre-filtered candidate genes from fast screen
    beacon_dir = Path(args.output_dir) / "beacon_edd"
    fast_path = beacon_dir / "fast_screen_mrna.csv"
    candidate_genes = None
    if fast_path.exists():
        fast_df = pd.read_csv(fast_path)
        candidate_genes = fast_df[fast_df["rho"] < -0.15]["gene"].tolist()
        log.info("Using %d candidate genes from fast screen for differential BEACON",
                 len(candidate_genes))

    diff_edd = differential_edd(
        depmap_expr, crispr, depmap_immune_status,
        group_a="immune_hot", group_b="immune_cold",
        method="beacon", candidate_genes=candidate_genes,
    )
    diff_edd.to_csv(out / "differential_edd_hot_vs_cold.csv", index=False)
    log.info("Differential EDD: %d genes with FDR < 0.05",
             (diff_edd["fdr"] < 0.05).sum() if not diff_edd.empty else 0)

    # ── E. Evasion-correlated EDD ──────────────────────────────────────
    log.info("=== Evasion-correlated EDD ===")
    # Use dependency as EDD proxy (filtered to high-expressors)
    ev_corr = evasion_correlated_edd(crispr, evasion_depmap)
    ev_corr.to_csv(out / "evasion_correlated_edd.csv", index=False)

    log.info("Immune context analysis complete.")


if __name__ == "__main__":
    main()
