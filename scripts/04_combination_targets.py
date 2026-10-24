#!/usr/bin/env python3
"""Step 04: Identify combination therapy targets via PRISM + druggability.

  A. Link EDD targets to PRISM drug-sensitivity profiles
  B. Query DGIdb for druggability annotations
  C. Score rational ICB combination candidates
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from combination.drug_sensitivity import (
    prism_target_sensitivity,
    query_dgidb,
    score_icb_combinations,
)
from data.depmap import load_expression, load_prism

log = get_logger("04_combination")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=CFG["data_dir"])
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    args = parser.parse_args()

    out = ensure_dir(Path(args.output_dir) / "combination")
    beacon_dir = Path(args.output_dir) / "beacon_edd"

    # Load significant EDD genes
    sig_path = beacon_dir / "beacon_edd_significant.csv"
    if not sig_path.exists():
        log.error("Run step 02 first: %s not found", sig_path)
        sys.exit(1)
    sig_edd = pd.read_csv(sig_path)
    edd_genes = sig_edd["gene"].unique().tolist()
    log.info("Starting with %d significant EDD genes", len(edd_genes))

    # ── A. PRISM drug sensitivity ──────────────────────────────────────
    log.info("=== PRISM drug-sensitivity analysis ===")
    expression = load_expression()
    prism_sens, drug_info = load_prism()

    prism_hits = prism_target_sensitivity(edd_genes, expression, prism_sens, drug_info)
    prism_hits.to_csv(out / "prism_edd_drug_hits.csv", index=False)
    log.info("PRISM hits: %d gene-drug pairs (FDR < 0.05)",
             (prism_hits["fdr"] < 0.05).sum() if not prism_hits.empty else 0)

    # ── B. Druggability annotation ─────────────────────────────────────
    log.info("=== DGIdb druggability query ===")
    dgidb = query_dgidb(edd_genes)
    dgidb.to_csv(out / "dgidb_annotations.csv", index=False)
    log.info("DGIdb: %d genes with drug interactions", dgidb["gene"].nunique() if not dgidb.empty else 0)

    # ── C. ICB combination scoring ─────────────────────────────────────
    log.info("=== Scoring ICB combination candidates ===")
    diff_edd_path = Path(args.output_dir) / "immune_context" / "differential_edd_hot_vs_cold.csv"
    if diff_edd_path.exists():
        diff_edd = pd.read_csv(diff_edd_path)
    else:
        diff_edd = pd.DataFrame()

    combos = score_icb_combinations(diff_edd, prism_hits, dgidb)
    combos.to_csv(out / "icb_combination_candidates.csv", index=False)
    log.info("Top combination candidates: %d genes scored", len(combos))

    log.info("Combination target discovery complete.")


if __name__ == "__main__":
    main()
