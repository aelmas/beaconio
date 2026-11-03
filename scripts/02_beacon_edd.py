#!/usr/bin/env python3
"""Step 02: Run BEACON expression-driven dependency analysis.

Runs BEACON (Bayesian + fast Spearman) across all lineages using DepMap
CRISPR dependency + expression/proteomics data.

Outputs:
  - Pan-lineage EDD results (gene x lineage)
  - Per-lineage significant EDD gene lists
  - Fast-screen results for pre-filtering
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.engine import beacon_fast, run_beacon_pan_lineage
from beacon_io.utils import ensure_dir, get_logger
from data.depmap import load_cell_line_info, load_crispr, load_expression, load_proteomics

log = get_logger("02_beacon")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=CFG["data_dir"])
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    parser.add_argument("--mode", choices=["fast", "full", "both"], default="both",
                        help="fast=Spearman only, full=Bayesian MCMC, both=screen then MCMC")
    parser.add_argument("--fast-rho-cutoff", type=float, default=-0.20,
                        help="Pre-filter cutoff for fast screen before full MCMC")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs for MCMC (-1 = all CPUs)")
    args = parser.parse_args()

    out = ensure_dir(Path(args.output_dir) / "beacon_edd")

    # Load data
    log.info("Loading DepMap data")
    crispr = load_crispr(args.data_dir)
    expression = load_expression(args.data_dir)
    cell_info = load_cell_line_info(args.data_dir)

    # ── Fast Spearman screen (all genes) ────────────────────────────────
    if args.mode in ("fast", "both"):
        log.info("Running fast Spearman screen (mRNA)")
        fast_mrna = beacon_fast(expression, crispr)
        fast_mrna.to_csv(out / "fast_screen_mrna.csv", index=False)
        log.info("Fast screen: %d genes with rho < %.2f",
                 (fast_mrna["rho"] < args.fast_rho_cutoff).sum(), args.fast_rho_cutoff)

        # Proteomics fast screen
        try:
            proteomics = load_proteomics(args.data_dir)
            fast_prot = beacon_fast(proteomics, crispr)
            fast_prot.to_csv(out / "fast_screen_protein.csv", index=False)
        except FileNotFoundError:
            log.warning("Proteomics data not available, skipping PED fast screen")

    # ── Full Bayesian BEACON (filtered genes) ───────────────────────────
    if args.mode in ("full", "both"):
        # Pre-filter to genes passing fast screen
        if args.mode == "both" and (out / "fast_screen_mrna.csv").exists():
            import pandas as pd
            fast = pd.read_csv(out / "fast_screen_mrna.csv")
            candidate_genes = fast[fast["rho"] < args.fast_rho_cutoff]["gene"].tolist()
            log.info("Pre-filtered to %d candidate genes for full MCMC", len(candidate_genes))
        else:
            candidate_genes = None

        log.info("Running full Bayesian BEACON MCMC across lineages (n_jobs=%s)", args.n_jobs)
        results = run_beacon_pan_lineage(
            expression, crispr, cell_info, genes=candidate_genes,
            n_jobs=args.n_jobs,
        )

        # Save per-lineage results
        for lineage, summary in results.items():
            df = summary.to_dataframe()
            safe_name = lineage.replace("/", "_").replace(" ", "_")
            df.to_csv(out / f"beacon_edd_{safe_name}.csv", index=False)

        # Compile pan-lineage summary
        import pandas as pd
        all_results = pd.concat(
            [s.to_dataframe() for s in results.values()],
            ignore_index=True,
        )
        all_results.to_csv(out / "beacon_edd_all_lineages.csv", index=False)

        # Also save the fast-screen results with a _spearman suffix for reference
        fast_all = out / "beacon_edd_all_lineages_spearman.csv"
        if not fast_all.exists() and (out / "fast_screen_mrna.csv").exists():
            import shutil
            # Back up the old Spearman all-lineages if it exists with old format
            old_all = out / "beacon_edd_all_lineages.csv.bak"
            if not old_all.exists():
                pass  # new MCMC results already written above

        sig = all_results[all_results["significant"]]
        sig.to_csv(out / "beacon_edd_significant.csv", index=False)
        log.info("Total significant EDD genes: %d (across %d lineages)",
                 sig["gene"].nunique(), sig["lineage"].nunique())

    log.info("BEACON EDD analysis complete.")


if __name__ == "__main__":
    main()
