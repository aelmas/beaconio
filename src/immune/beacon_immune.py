"""BEACON-IO: immune-contextualized expression-driven dependencies.

Key innovation over original BEACON:
  Rather than running BEACON uniformly, we stratify cell lines (or tumour
  samples) by immune context and identify dependencies that are:

  1. **Immune-specific EDD**: significant in immune-hot but not immune-cold
     tumours (or vice versa) — targets that become vulnerabilities in
     specific immune microenvironments.

  2. **Immune-evasion EDD**: genes whose EDD is correlated with immune
     evasion programmes — targets whose dependency scales with the
     degree of immune escape.

  3. **ICB-resistance EDD**: dependencies enriched among ICB non-responders
     vs responders — directly actionable combination targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from beacon_io.engine import BeaconSummary, beacon_fast, run_beacon_lineage
from beacon_io.utils import fdr_correction, get_logger

log = get_logger(__name__)


def differential_edd(
    expression: pd.DataFrame,
    dependency: pd.DataFrame,
    stratification: pd.Series,
    group_a: str = "immune_hot",
    group_b: str = "immune_cold",
    method: str = "beacon",
    candidate_genes: list[str] | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Identify genes with differential expression-driven dependency
    between two immune strata using full Bayesian BEACON.

    For each gene, runs BEACON MCMC in group_a and group_b separately,
    then compares posteriors: overlap of HDIs + Fisher z on posterior medians.

    Parameters
    ----------
    method : 'beacon' for full MCMC (default), 'fast' for Spearman fallback.
    candidate_genes : pre-filtered gene list to limit MCMC to viable candidates.
    n_jobs : parallel jobs for BEACON MCMC (-1 = all CPUs).

    Returns DataFrame with columns:
      gene, rho_{group_a}, rho_{group_b}, delta_rho, pvalue, fdr,
      hdi_low_a, hdi_high_a, hdi_low_b, hdi_high_b, prob_neg_a, prob_neg_b
    """
    mask_a = stratification == group_a
    mask_b = stratification == group_b

    common = expression.index.intersection(dependency.index)
    ids_a = common[common.isin(stratification[mask_a].index)]
    ids_b = common[common.isin(stratification[mask_b].index)]

    if len(ids_a) < 5 or len(ids_b) < 5:
        log.warning("Too few samples for differential EDD (A=%d, B=%d)", len(ids_a), len(ids_b))
        return pd.DataFrame()

    shared_genes = expression.columns.intersection(dependency.columns)
    if candidate_genes:
        shared_genes = shared_genes.intersection(candidate_genes)

    if method == "beacon":
        from beacon_io.engine import run_beacon_lineage

        log.info("Running BEACON MCMC for differential EDD: %d genes, group_a=%d, group_b=%d",
                 len(shared_genes), len(ids_a), len(ids_b))

        # Run BEACON on each group
        summary_a = run_beacon_lineage(
            expression.loc[ids_a], dependency.loc[ids_a],
            lineage=group_a, genes=list(shared_genes), n_jobs=n_jobs,
        )
        summary_b = run_beacon_lineage(
            expression.loc[ids_b], dependency.loc[ids_b],
            lineage=group_b, genes=list(shared_genes), n_jobs=n_jobs,
        )

        # Index results by gene
        res_a = {r.gene: r for r in summary_a.results}
        res_b = {r.gene: r for r in summary_b.results}

        records = []
        for gene in set(res_a.keys()) & set(res_b.keys()):
            ra, rb = res_a[gene], res_b[gene]
            # Fisher z-test on posterior medians
            z_a = np.arctanh(np.clip(ra.rho_posterior_median, -0.999, 0.999))
            z_b = np.arctanh(np.clip(rb.rho_posterior_median, -0.999, 0.999))
            se = np.sqrt(1 / (ra.n_samples - 3) + 1 / (rb.n_samples - 3))
            z_diff = (z_a - z_b) / se
            pval = 2 * stats.norm.sf(abs(z_diff))

            records.append({
                "gene": gene,
                f"rho_{group_a}": ra.rho_posterior_median,
                f"rho_{group_b}": rb.rho_posterior_median,
                "delta_rho": ra.rho_posterior_median - rb.rho_posterior_median,
                "z_fisher": z_diff,
                "pvalue": pval,
                "hdi_low_a": ra.rho_hdi_low,
                "hdi_high_a": ra.rho_hdi_high,
                "hdi_low_b": rb.rho_hdi_low,
                "hdi_high_b": rb.rho_hdi_high,
                "prob_neg_a": ra.prob_negative,
                "prob_neg_b": rb.prob_negative,
                "ess_a": ra.ess_bulk,
                "ess_b": rb.ess_bulk,
                "rhat_a": ra.rhat,
                "rhat_b": rb.rhat,
            })
    else:
        # Spearman fallback
        records = []
        for gene in shared_genes:
            e_a, d_a = expression.loc[ids_a, gene].values, dependency.loc[ids_a, gene].values
            e_b, d_b = expression.loc[ids_b, gene].values, dependency.loc[ids_b, gene].values
            if e_a.std() < 1e-8 or d_a.std() < 1e-8 or e_b.std() < 1e-8 or d_b.std() < 1e-8:
                continue
            rho_a, _ = stats.spearmanr(e_a, d_a)
            rho_b, _ = stats.spearmanr(e_b, d_b)
            z_a = np.arctanh(np.clip(rho_a, -0.999, 0.999))
            z_b = np.arctanh(np.clip(rho_b, -0.999, 0.999))
            se = np.sqrt(1 / (len(ids_a) - 3) + 1 / (len(ids_b) - 3))
            z_diff = (z_a - z_b) / se
            pval = 2 * stats.norm.sf(abs(z_diff))
            records.append({
                "gene": gene,
                f"rho_{group_a}": rho_a,
                f"rho_{group_b}": rho_b,
                "delta_rho": rho_a - rho_b,
                "z_fisher": z_diff,
                "pvalue": pval,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["fdr"] = fdr_correction(df["pvalue"].values)
        df = df.sort_values("pvalue")
    return df


def evasion_correlated_edd(
    edd_scores: pd.DataFrame,
    evasion_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate per-gene EDD strength with immune evasion programme scores.

    For each (gene, evasion_programme) pair, compute Spearman correlation
    between the gene's expression-dependency profile and the evasion score
    across cell lines / samples.

    Parameters
    ----------
    edd_scores : DataFrame (samples x genes) of expression * dependency product
                 (or just dependency scores filtered to high-expression samples).
    evasion_scores : DataFrame (samples x evasion_programmes).

    Returns
    -------
    DataFrame with gene, programme, rho, pvalue, fdr.
    """
    shared = edd_scores.index.intersection(evasion_scores.index)
    records = []
    for gene in edd_scores.columns:
        g = edd_scores.loc[shared, gene].values
        if np.std(g) < 1e-8:
            continue
        for prog in evasion_scores.columns:
            p = evasion_scores.loc[shared, prog].values
            if np.std(p) < 1e-8:
                continue
            rho, pval = stats.spearmanr(g, p)
            records.append({
                "gene": gene,
                "evasion_programme": prog,
                "rho": rho,
                "pvalue": pval,
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df["fdr"] = fdr_correction(df["pvalue"].values)
    return df


def icb_response_edd(
    expression: pd.DataFrame,
    dependency_predictions: pd.DataFrame,
    response: pd.Series,
) -> pd.DataFrame:
    """Identify EDD genes enriched among ICB responders vs non-responders.

    Since patient tumours lack CRISPR dependency data, we use predicted
    dependency from BEACON's expression-dependency model (trained on cell
    lines) applied to patient expression profiles.

    Parameters
    ----------
    expression : patient expression (samples x genes).
    dependency_predictions : predicted dependency scores (samples x genes).
    response : binary response (1 = responder, 0 = non-responder).

    Returns
    -------
    DataFrame with gene, edd_responders, edd_nonresponders, delta, pvalue, fdr.
    """
    shared = expression.index.intersection(dependency_predictions.index).intersection(response.index)
    resp_ids = response[response == 1].index.intersection(shared)
    nonresp_ids = response[response == 0].index.intersection(shared)

    records = []
    for gene in expression.columns.intersection(dependency_predictions.columns):
        # EDD product = expression * dependency (more negative = stronger dependency in high expressors)
        edd_r = (
            expression.loc[resp_ids, gene].values
            * dependency_predictions.loc[resp_ids, gene].values
        )
        edd_nr = (
            expression.loc[nonresp_ids, gene].values
            * dependency_predictions.loc[nonresp_ids, gene].values
        )
        if len(edd_r) < 3 or len(edd_nr) < 3:
            continue
        stat, pval = stats.mannwhitneyu(edd_r, edd_nr, alternative="two-sided")
        records.append({
            "gene": gene,
            "edd_responders": np.median(edd_r),
            "edd_nonresponders": np.median(edd_nr),
            "delta_edd": np.median(edd_nr) - np.median(edd_r),
            "U_statistic": stat,
            "pvalue": pval,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["fdr"] = fdr_correction(df["pvalue"].values)
    return df
