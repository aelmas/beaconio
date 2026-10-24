#!/usr/bin/env python3
"""Step 06: Single-cell resolution of BEACON-IO targets.

  A. Load scRNA-seq atlases (melanoma ICB, pan-cancer T cell)
  B. Cell-type expression profiling of BEACON-IO targets
  C. Compartment classification (tumour vs immune vs stromal)
  D. Differential expression of targets in responders vs non-responders
  E. Evasion programme scoring at single-cell level
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from singlecell.resolution import (
    celltype_expression_profile,
    classify_target_compartment,
    load_singlecell_atlas,
    sc_differential_beacon_targets,
    score_evasion_programmes_sc,
)

log = get_logger("06_singlecell")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=CFG["data_dir"])
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    args = parser.parse_args()

    out = ensure_dir(Path(args.output_dir) / "singlecell")
    beacon_dir = Path(args.output_dir) / "beacon_edd"

    # Load BEACON-IO significant genes
    sig_path = beacon_dir / "beacon_edd_significant.csv"
    sig_edd = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    beacon_genes = sig_edd["gene"].unique().tolist() if not sig_edd.empty else []

    if not beacon_genes:
        log.warning("No BEACON-IO genes found. Run step 02 first.")
        return

    all_compartments = []

    for sc_cfg in CFG["singlecell"]:
        name = sc_cfg["name"]
        log.info("=== Processing %s ===", name)

        adata = load_singlecell_atlas(name)
        if adata.n_obs == 0:
            log.warning("Skipping %s (not available)", name)
            continue

        # ── B. Cell-type expression profiles ───────────────────────────
        ct_expr = celltype_expression_profile(adata, beacon_genes)
        if not ct_expr.empty:
            ct_expr.to_csv(out / f"{name}_celltype_expression.csv")

        # ── C. Compartment classification ──────────────────────────────
        compartment = classify_target_compartment(ct_expr)
        if not compartment.empty:
            compartment["dataset"] = name
            compartment.to_csv(out / f"{name}_compartment.csv", index=False)
            all_compartments.append(compartment)

        # ── D. Differential expression R vs NR ─────────────────────────
        if "response" in adata.obs.columns:
            diff = sc_differential_beacon_targets(adata, beacon_genes)
            if not diff.empty:
                diff.to_csv(out / f"{name}_diff_expr_R_vs_NR.csv", index=False)

        # ── E. Evasion programme scoring ───────────────────────────────
        adata = score_evasion_programmes_sc(adata)
        # Save programme scores
        prog_cols = [c for c in adata.obs.columns if c in [
            "antigen_presentation_loss", "ifn_gamma_signaling",
            "tgfb_exclusion", "wnt_beta_catenin", "immune_checkpoints",
            "myeloid_suppression",
        ]]
        if prog_cols:
            adata.obs[prog_cols].to_csv(out / f"{name}_evasion_scores.csv")

    # Merge compartment classifications across datasets
    if all_compartments:
        merged = pd.concat(all_compartments, ignore_index=True)
        # Consensus: majority vote across datasets
        consensus = (
            merged.groupby("gene")["primary_compartment"]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
        )
        consensus.columns = ["gene", "consensus_compartment"]
        consensus.to_csv(out / "compartment_consensus.csv", index=False)
        log.info("Compartment consensus: %s",
                 consensus["consensus_compartment"].value_counts().to_dict())

    log.info("Single-cell analysis complete.")


if __name__ == "__main__":
    main()
