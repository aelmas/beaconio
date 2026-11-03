#!/usr/bin/env python3
"""Step 03e: Within-lineage permutation null for the differential EDD screen.

The differential EDD screen compares each candidate gene's expression-dependency
correlation between immune-hot and immune-cold cell lines. To calibrate how
often such a screen flags genes by chance, we permute the immune-hot/cold
labels *within each tumour-lineage stratum* B times and rerun the screen.
Every permutation therefore reproduces the observed lineage sizes and
lineage-wise hot/cold composition exactly, and the resulting null tests
immune-label association beyond lineage structure, not merely exchangeability.
We record (i) the number of genes passing BH FDR < 0.05 per permutation and
(ii) per-gene empirical two-sided permutation p-values, computed as
(b + 1) / (B + 1).

The screen statistic family matches the BEACON MCMC screen: per-gene Pearson
correlations (BEACON fits bivariate-normal likelihoods, whose MLE correlation is
Pearson) combined through Fisher's z. NaN dependency values are handled by
pairwise deletion per gene, as in the production screen.

Outputs
-------
analysis/out/{immune_context,immune_context_solid}/permutation_null_counts.csv
    Per-permutation counts of BH-significant genes in the permuted screens.
analysis/out/{immune_context,immune_context_solid}/permutation_null_genes.csv
    Per-gene observed z, BH FDR, and empirical permutation p.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from data.depmap import load_cell_line_info, load_crispr, load_expression

log = get_logger("03e_permutation_null")

N_PERM = 300
SEED = 42


def bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def _fisher_z_diff(x: np.ndarray, d: np.ndarray, hot_m: np.ndarray,
                   cold_m: np.ndarray) -> float:
    """Pearson r in the two subsets (pairwise deletion) + Fisher z difference."""
    ra, _ = stats.pearsonr(x[hot_m], d[hot_m])
    rb, _ = stats.pearsonr(x[cold_m], d[cold_m])
    if np.isnan(ra) or np.isnan(rb):
        return np.nan
    za = np.arctanh(np.clip(ra, -0.999, 0.999))
    zb = np.arctanh(np.clip(rb, -0.999, 0.999))
    se = np.sqrt(1 / (hot_m.sum() - 3) + 1 / (cold_m.sum() - 3))
    return (za - zb) / se


def _batch_fisher_z(x: np.ndarray, d: np.ndarray, valid: np.ndarray,
                    hot_mats: tuple[np.ndarray, np.ndarray],
                    cold_mats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Exact Fisher-z of the Pearson difference for every permutation row.

    ``hot_mats``/``cold_mats`` are (weight-matrix, count-vector) pairs over the
    valid rows; the weight matrices are 0/1 membership masks, one row per
    permutation. Pearson within each subset uses the subset's own mean/variance.
    """
    hw, hn = hot_mats
    cw, cn = cold_mats
    xy = x * d
    x2 = x * x
    y2 = d * d
    sxy_h, sx_h, sy_h, sx2_h, sy2_h = (hw @ xy, hw @ x, hw @ d,
                                       hw @ x2, hw @ y2)
    sxy_c, sx_c, sy_c, sx2_c, sy2_c = (cw @ xy, cw @ x, cw @ d,
                                       cw @ x2, cw @ y2)
    cov_h = hn * sxy_h - sx_h * sy_h
    var_x_h = hn * sx2_h - sx_h ** 2
    var_y_h = hn * sy2_h - sy_h ** 2
    cov_c = cn * sxy_c - sx_c * sy_c
    var_x_c = cn * sx2_c - sx_c ** 2
    var_y_c = cn * sy2_c - sy_c ** 2
    ok = (var_x_h > 0) & (var_y_h > 0) & (var_x_c > 0) & (var_y_c > 0)
    ra = np.where(ok, cov_h / np.sqrt(np.where(var_x_h * var_y_h > 0,
                                              var_x_h * var_y_h, 1)), np.nan)
    rb = np.where(ok, cov_c / np.sqrt(np.where(var_x_c * var_y_c > 0,
                                              var_x_c * var_y_c, 1)), np.nan)
    za = np.arctanh(np.clip(ra, -0.999, 0.999))
    zb = np.arctanh(np.clip(rb, -0.999, 0.999))
    se = np.sqrt(1 / np.maximum(hn - 3, 1) + 1 / np.maximum(cn - 3, 1))
    return (za - zb) / se


def run_analysis(ids_common: pd.Index, status: pd.Series, cand_genes: list[str],
                 expr_df: pd.DataFrame, dep_df: pd.DataFrame, tag: str,
                 out_dir: Path, rng: np.random.Generator,
                 lineage: pd.Series | None = None) -> None:
    """Run one permutation analysis (full cohort or solid strata).

    ``lineage`` (optional) enables *within-lineage* label permutation: hot/cold
    labels are reshuffled only among cell lines of the same OncotreeLineage,
    so every permutation reproduces the observed lineage composition exactly.
    This null tests immune-label association beyond lineage structure.
    """
    labels = status.reindex(ids_common).to_numpy()
    mask_a = labels == "immune_hot"
    mask_b = ~mask_a
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    log.info("[%s] paired lines: %d (hot %d / cold %d)", tag, len(ids_common),
             n_a, n_b)

    # Within-lineage permutation: precompute lineage groups and, per
    # permutation, reshuffle labels only within each group.
    if lineage is not None:
        lin = lineage.reindex(ids_common).to_numpy()
        groups = [idx for idx in
                  [np.where(lin == lab)[0] for lab in pd.unique(lin)]
                  if len(idx) > 1]
        log.info("[%s] within-lineage permutation over %d lineage groups",
                 tag, len(groups))
    base = mask_a.copy()

    def permute() -> np.ndarray:
        arr = base.copy()
        if lineage is None:
            sel = arr[mask_a | mask_b]
            rng.shuffle(sel)
            arr[mask_a | mask_b] = sel
            return arr
        for g in groups:
            vals = arr[g]
            # keep each group's own hot/cold counts: shuffle its labels
            n_keep = int(vals.sum())
            vals = np.zeros(len(g), dtype=bool)
            vals[rng.choice(len(g), size=n_keep, replace=False)] = True
            arr[g] = vals
        return arr

    # Permutation membership matrices (0/1), one row per permutation
    hot_mats, cold_mats = [], []
    for _ in range(N_PERM):
        perm = permute()
        hot_mats.append(perm.astype(float))
        cold_mats.append((~perm).astype(float))
    hot_w = np.vstack(hot_mats)
    cold_w = np.vstack(cold_mats)

    z_null = np.empty((N_PERM, len(cand_genes)))
    z_obs = np.empty(len(cand_genes))
    for j, gene in enumerate(cand_genes):
        d_full = dep_df[gene].reindex(ids_common).to_numpy(dtype=float)
        valid = ~np.isnan(d_full)
        valid &= ~np.isnan(expr_df[gene].reindex(ids_common).to_numpy(float))
        x_all = expr_df[gene].reindex(ids_common).to_numpy(dtype=float)[valid]
        d_all = d_full[valid]

        hm, cm = hot_w[:, valid], cold_w[:, valid]
        hot = (hm, hm.sum(axis=1))
        cold = (cm, cm.sum(axis=1))
        z_null[:, j] = _batch_fisher_z(x_all, d_all, valid, hot, cold)

        hot_m = (mask_a & valid)
        cold_m = (mask_b & valid)
        z_obs[j] = _fisher_z_diff(
            expr_df[gene].reindex(ids_common).to_numpy(dtype=float),
            d_full, hot_m, cold_m)

    p_obs = 2 * stats.norm.sf(np.abs(z_obs))
    fdr_obs = bh(p_obs)
    # (b+1)/(m+1) resolution: never a zero p under finite permutations
    perm_p = (np.nansum(np.abs(z_null) >= np.abs(z_obs), axis=0) + 1) / (N_PERM + 1)

    counts_df = pd.DataFrame({
        "perm": range(N_PERM),
        "n_fdr_sig": [int((bh(2 * stats.norm.sf(np.abs(z_null[b]))) < 0.05)
                          .sum())
                      for b in range(N_PERM)],
    })
    counts_df.to_csv(out_dir / f"permutation_null_counts{'_' + tag}.csv",
                     index=False)

    out = pd.DataFrame({
        "gene": cand_genes,
        "z_fisher_obs": z_obs,
        "fisher_fdr_obs": fdr_obs,
        "fisher_sig_obs": fdr_obs < 0.05,
        "perm_p_two_sided": perm_p,
    })
    out.to_csv(out_dir / f"permutation_null_genes{'_' + tag}.csv", index=False)

    ok_obs = int((fdr_obs < 0.05).sum())
    log.info("[%s] observed BH-significant: %d | null median: %.1f, "
             "95th pct: %.1f", tag, ok_obs, counts_df["n_fdr_sig"].median(),
             counts_df["n_fdr_sig"].quantile(0.95))


def main() -> None:
    rng = np.random.default_rng(SEED)
    out_solid = ensure_dir(Path(CFG["output_dir"]) / "immune_context_solid")
    out_full = ensure_dir(Path(CFG["output_dir"]) / "immune_context")

    log.info("Loading DepMap data")
    expr = load_expression()
    dep = load_crispr()

    # Candidate genes: the 441 differential candidates (rho < -0.15 screen)
    fast = pd.read_csv(Path(CFG["output_dir"]) / "beacon_edd" /
                       "fast_screen_mrna.csv")
    cand = fast[fast["rho"] < -0.15]["gene"].tolist()
    cand = [g for g in cand if g in expr.columns and g in dep.columns]
    log.info("Candidate genes: %d", len(cand))

    # OncotreeLineage per cell line, for within-lineage label permutation
    cell_info = load_cell_line_info()
    lineage = cell_info["PrimaryDisease"]

    common = expr.index.intersection(dep.index)

    # --- Full-cohort screen (24 lineages) ---
    full_status = pd.read_csv(Path(CFG["output_dir"]) / "immune_context" /
                              "depmap_immune_status.csv",
                              index_col=0).iloc[:, 0]
    run_analysis(common.intersection(full_status.index), full_status, cand,
                 expr, dep, "full", out_full, rng, lineage)

    # --- Solid strata (hematopoietic excluded; baked into the saved file) ---
    solid_status = pd.read_csv(Path(CFG["output_dir"]) / "immune_context_solid" /
                               "solid_immune_status.csv",
                               index_col=0).iloc[:, 0]
    run_analysis(common.intersection(solid_status.index), solid_status, cand,
                 expr, dep, "solid", out_solid, rng, lineage)

    log.info("Done.")


if __name__ == "__main__":
    main()
