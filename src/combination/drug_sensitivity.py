"""Link BEACON-IO targets to drug sensitivity (PRISM) and druggability databases.

Pipeline:
  1. Map EDD targets to PRISM drug-sensitivity profiles.
  2. Identify drugs whose sensitivity correlates with expression of EDD targets.
  3. Cross-reference with DGIdb / DrugBank for druggability annotation.
  4. Score rational ICB combination candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

from beacon_io.config import CFG
from beacon_io.utils import fdr_correction, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# PRISM drug-sensitivity analysis
# ---------------------------------------------------------------------------

def prism_target_sensitivity(
    edd_genes: list[str],
    expression: pd.DataFrame,
    prism_sensitivity: pd.DataFrame,
    drug_info: pd.DataFrame,
) -> pd.DataFrame:
    """For each EDD target, find drugs whose sensitivity correlates with
    that gene's expression across cell lines.

    High expression of EDD gene + high sensitivity to drug X = candidate
    combination (drug X targets the same dependency the tumour relies on).

    Returns DataFrame: gene, drug, drug_name, target, rho, pvalue, fdr.
    """
    shared_lines = expression.index.intersection(prism_sensitivity.index)
    log.info(
        "PRISM-expression overlap: %d cell lines, testing %d EDD genes x %d drugs",
        len(shared_lines), len(edd_genes), prism_sensitivity.shape[1],
    )

    records = []
    for gene in edd_genes:
        if gene not in expression.columns:
            continue
        expr_vals = expression.loc[shared_lines, gene].values
        if expr_vals.std() < 1e-8:
            continue
        for drug_col in prism_sensitivity.columns:
            sens_vals = prism_sensitivity.loc[shared_lines, drug_col].values
            valid = ~(np.isnan(expr_vals) | np.isnan(sens_vals))
            if valid.sum() < 10:
                continue
            rho, pval = stats.spearmanr(expr_vals[valid], sens_vals[valid])
            # Negative rho = high expression -> more negative log-fold (more sensitive)
            if rho < -0.2 and pval < 0.05:
                records.append({
                    "gene": gene,
                    "drug_id": drug_col,
                    "rho": rho,
                    "pvalue": pval,
                    "n_lines": int(valid.sum()),
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["fdr"] = fdr_correction(df["pvalue"].values)

    # Merge drug annotations
    if "column_name" in drug_info.columns:
        df = df.merge(
            drug_info[["column_name", "name", "target", "moa", "phase"]].drop_duplicates(),
            left_on="drug_id",
            right_on="column_name",
            how="left",
        )
    return df.sort_values("rho")


# ---------------------------------------------------------------------------
# Druggability annotation via DGIdb
# ---------------------------------------------------------------------------

def query_dgidb(genes: list[str]) -> pd.DataFrame:
    """Query DGIdb API for drug-gene interactions.

    Returns DataFrame: gene, drug_name, interaction_type, source.
    """
    url = "https://dgidb.org/api/v2/interactions.json"
    records = []
    # Batch in chunks of 100
    for i in range(0, len(genes), 100):
        batch = genes[i : i + 100]
        resp = requests.get(url, params={"genes": ",".join(batch)}, timeout=60)
        if not resp.ok:
            log.warning("DGIdb query failed: %s", resp.status_code)
            continue
        data = resp.json()
        for match in data.get("matchedTerms", []):
            gene = match["geneName"]
            for interaction in match.get("interactions", []):
                records.append({
                    "gene": gene,
                    "drug_name": interaction.get("drugName", ""),
                    "interaction_type": interaction.get("interactionTypes", ""),
                    "source": interaction.get("sources", ""),
                    "pmid": interaction.get("pmids", ""),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Combination scoring
# ---------------------------------------------------------------------------

def score_icb_combinations(
    edd_immune_df: pd.DataFrame,
    prism_hits: pd.DataFrame,
    dgidb_hits: pd.DataFrame,
) -> pd.DataFrame:
    """Score and rank candidate ICB combination targets.

    Combines evidence from:
      - Immune-specific EDD strength (from differential_edd)
      - PRISM drug-sensitivity correlation
      - Druggability (DGIdb annotations)

    Returns ranked DataFrame of gene-drug combinations with composite score.
    """
    # Start from EDD genes with immune-specific signal
    if edd_immune_df.empty:
        return pd.DataFrame()

    combo = edd_immune_df.copy()

    # Add PRISM evidence
    if not prism_hits.empty:
        best_prism = (
            prism_hits.groupby("gene")
            .agg(best_drug=("name", "first"), prism_rho=("rho", "min"), prism_fdr=("fdr", "min"))
            .reset_index()
        )
        combo = combo.merge(best_prism, on="gene", how="left")

    # Add druggability
    if not dgidb_hits.empty:
        druggable = dgidb_hits.groupby("gene").agg(
            n_drugs=("drug_name", "nunique"),
            drug_list=("drug_name", lambda x: "; ".join(x.unique()[:5])),
        ).reset_index()
        combo = combo.merge(druggable, on="gene", how="left")
        combo["is_druggable"] = combo["n_drugs"].fillna(0) > 0
    else:
        combo["is_druggable"] = False

    # Composite score: rank-based combination of EDD strength + drug sensitivity + druggability
    combo["rank_edd"] = combo["delta_rho"].abs().rank(ascending=False, pct=True) \
        if "delta_rho" in combo.columns else 0.5
    combo["rank_prism"] = combo["prism_rho"].abs().rank(ascending=False, pct=True) \
        if "prism_rho" in combo.columns else 0.5
    combo["rank_druggable"] = combo["is_druggable"].astype(float)

    combo["combo_score"] = (
        0.4 * combo["rank_edd"]
        + 0.35 * combo["rank_prism"]
        + 0.25 * combo["rank_druggable"]
    )

    return combo.sort_values("combo_score", ascending=False)
