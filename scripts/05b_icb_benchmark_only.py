#!/usr/bin/env python3
"""Quick re-run of just the ICB benchmark portion (skips TCGA Cox PH).

Used to regenerate icb_biomarker_benchmark.csv after changing
benchmark_biomarkers() in src/clinical/validation.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from clinical.validation import benchmark_biomarkers, build_beacon_io_signature
from data.icb_cohorts import load_hugo_2016

log = get_logger("05b_icb_only")


def main():
    out = ensure_dir(Path(CFG["output_dir"]) / "clinical")
    beacon_dir = Path(CFG["output_dir"]) / "beacon_edd"
    immune_dir = Path(CFG["output_dir"]) / "immune_context"

    sig_path = beacon_dir / "beacon_edd_significant.csv"
    sig_edd = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    beacon_genes = sig_edd["gene"].unique().tolist() if not sig_edd.empty else []

    diff_path = immune_dir / "differential_edd_hot_vs_cold.csv"
    if diff_path.exists():
        diff_edd = pd.read_csv(diff_path)
        signature_genes = build_beacon_io_signature(diff_edd, top_n=50)
    else:
        signature_genes = beacon_genes[:50]

    log.info("BEACON-IO signature: %d genes", len(signature_genes))

    # Load only Hugo 2016 (the Mariathasan/Braun loaders have unrelated bugs
    # and the main step 05 has the same data; we only need Hugo for Fig 5).
    data_dir = Path(CFG["data_dir"])
    icb_cohorts = {"Hugo_2016": load_hugo_2016(data_dir)}
    cohort_benchmarks = []

    for name, cohort in icb_cohorts.items():
        expr = cohort["expression"]
        clin = cohort["clinical"]
        if expr.empty or "response_binary" not in clin.columns:
            continue
        response = clin["response_binary"]
        log.info("  %s: n=%d (R=%d, NR=%d)", name, len(response),
                 int(response.sum()), int(len(response) - response.sum()))
        bench = benchmark_biomarkers(expr, clin, response, signature_genes)
        bench["cohort"] = name
        cohort_benchmarks.append(bench)

    if cohort_benchmarks:
        bench_df = pd.concat(cohort_benchmarks, ignore_index=True)
        bench_df.to_csv(out / "icb_biomarker_benchmark.csv", index=False)
        log.info("Updated icb_biomarker_benchmark.csv:\n%s", bench_df.to_string(index=False))


if __name__ == "__main__":
    main()
