#!/usr/bin/env python3
"""Step 05: Clinical validation against ICB cohorts and TCGA survival.

  A. TCGA survival analysis (per cancer type)
  B. ICB response prediction (BEACON-IO vs existing biomarkers)
  C. Multi-cohort meta-analysis
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from clinical.validation import (
    benchmark_biomarkers,
    build_beacon_io_signature,
    meta_analysis_icb,
    predict_icb_response,
    tcga_survival_analysis,
)
from data.icb_cohorts import load_all_icb_cohorts
from data.tcga import load_tcga_expression, load_tcga_survival

log = get_logger("05_clinical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=CFG["data_dir"])
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    args = parser.parse_args()

    out = ensure_dir(Path(args.output_dir) / "clinical")
    beacon_dir = Path(args.output_dir) / "beacon_edd"

    # Load BEACON-IO results
    sig_path = beacon_dir / "beacon_edd_significant.csv"
    sig_edd = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    beacon_genes = sig_edd["gene"].unique().tolist() if not sig_edd.empty else []

    # Build BEACON-IO signature (top 50 genes)
    # Merge with immune-context results if available
    immune_dir = Path(args.output_dir) / "immune_context"
    diff_path = immune_dir / "differential_edd_hot_vs_cold.csv"
    if diff_path.exists():
        diff_edd = pd.read_csv(diff_path)
        signature_genes = build_beacon_io_signature(diff_edd, top_n=50)
    else:
        signature_genes = beacon_genes[:50]

    log.info("BEACON-IO signature: %d genes", len(signature_genes))

    # ── A. TCGA survival analysis ──────────────────────────────────────
    log.info("=== TCGA survival analysis ===")
    tcga_survival_results = []
    for cancer in CFG["tcga"]["cancer_types"]:
        log.info("  %s", cancer)
        try:
            expr = load_tcga_expression(cancer_types=[cancer])
            surv = load_tcga_survival()
            result = tcga_survival_analysis(expr, surv, beacon_genes, cancer_type=cancer)
            if not result.empty:
                tcga_survival_results.append(result)
        except Exception as exc:
            log.warning("Failed for %s: %s", cancer, exc)

    if tcga_survival_results:
        tcga_surv = pd.concat(tcga_survival_results, ignore_index=True)
        tcga_surv.to_csv(out / "tcga_survival_results.csv", index=False)
        sig_surv = tcga_surv[tcga_surv["cox_fdr"] < 0.05]
        log.info("TCGA survival: %d gene-cancer pairs with FDR < 0.05", len(sig_surv))

    # ── B. ICB response prediction ─────────────────────────────────────
    log.info("=== ICB response prediction ===")
    icb_cohorts = load_all_icb_cohorts()
    cohort_aucs = []
    cohort_benchmarks = []
    cohort_rho_results = {}

    for name, cohort in icb_cohorts.items():
        expr = cohort["expression"]
        clin = cohort["clinical"]
        if expr.empty or "response_binary" not in clin.columns:
            log.warning("Skipping %s (missing data)", name)
            continue

        response = clin["response_binary"]
        log.info("  %s: n=%d (R=%d, NR=%d)", name, len(response),
                 response.sum(), len(response) - response.sum())

        # BEACON-IO prediction
        res = predict_icb_response(expr, response, signature_genes)
        res["cohort"] = name
        cohort_aucs.append(res)

        # Benchmark against existing biomarkers
        bench = benchmark_biomarkers(
            expr, clin, response, signature_genes,
        )
        bench["cohort"] = name
        cohort_benchmarks.append(bench)

    if cohort_aucs:
        auc_df = pd.DataFrame(cohort_aucs)
        auc_df.to_csv(out / "icb_response_aucs.csv", index=False)
        log.info("ICB prediction AUCs:\n%s",
                 auc_df[["cohort", "auc_mean", "n_genes"]].to_string(index=False))

    if cohort_benchmarks:
        bench_df = pd.concat(cohort_benchmarks, ignore_index=True)
        bench_df.to_csv(out / "icb_biomarker_benchmark.csv", index=False)

    # ── C. Meta-analysis ──────────────────────────────────────────────
    log.info("=== ICB meta-analysis ===")
    if cohort_rho_results:
        meta = meta_analysis_icb(cohort_rho_results)
        meta.to_csv(out / "icb_meta_analysis.csv", index=False)

    log.info("Clinical validation complete.")


if __name__ == "__main__":
    main()
