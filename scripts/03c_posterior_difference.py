#!/usr/bin/env python3
"""Step 03c: Coherent posterior-difference test for differential EDD (addresses
referee concern C1).

The original differential test applied a frequentist Fisher-z transform to the
posterior MEDIANS of rho_hot and rho_cold, plugging in the full group n as the
SE denominator. That discards the MCMC posterior uncertainty and is neither a
valid Bayesian nor a valid frequentist test.

Here we test the difference directly on the posteriors BEACON already samples:
for each gene we run BEACON MCMC in the immune-hot and immune-cold solid-tumor
strata, retain the rho posterior draws, form the difference distribution
  Delta_rho = rho_hot - rho_cold      (independent groups -> element-wise
                                       difference of the two posterior vectors)
and report:
  - P(Delta_rho < 0 | data)   : posterior prob. dependency is STRONGER in hot
  - median and 95% HDI of Delta_rho
  - posterior_significant      : 95% HDI of Delta_rho excludes 0
                                 (equivalently P(Delta_rho<0) > 0.975)

We run this for the genes the Fisher-z analysis flagged FDR<0.05 so the two
tests can be compared head-to-head.

Output: analysis/out/immune_context_solid/posterior_difference_solid.csv
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from data.depmap import load_cell_line_info, load_crispr, load_expression
from immune.deconvolution import run_estimate, stratify_immune

log = get_logger("03c_postdiff")

HEME = {"Lymphoid", "Myeloid"}
BCFG = CFG["beacon"]


def _rho_draws(expr, dep):
    """Run the BEACON bivariate-normal model and return flattened rho draws."""
    if expr.std() < 1e-8 or dep.std() < 1e-8 or len(expr) < 10:
        return None
    ez = (expr - expr.mean()) / (expr.std() + 1e-8)
    dz = (dep - dep.mean()) / (dep.std() + 1e-8)
    obs = np.column_stack([ez, dz])
    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=2, shape=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5, shape=2)
        rho = pm.Uniform("rho", lower=-1, upper=1)
        cov = pm.math.stack([
            [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1] ** 2],
        ])
        pm.MvNormal("obs", mu=mu, cov=cov, observed=obs)
        trace = pm.sample(
            draws=BCFG["n_draws"], tune=BCFG["n_tune"], chains=BCFG["n_chains"],
            cores=1, target_accept=BCFG["target_accept"],
            return_inferencedata=True, progressbar=False, random_seed=42,
        )
    return trace.posterior["rho"].values.flatten()


def _posterior_diff_gene(eg_hot, dg_hot, eg_cold, dg_cold, gene):
    try:
        rh = _rho_draws(eg_hot, dg_hot)
        rc = _rho_draws(eg_cold, dg_cold)
        if rh is None or rc is None:
            return None
        # Independent groups: pair draws element-wise (truncate to common len)
        m = min(len(rh), len(rc))
        delta = rh[:m] - rc[:m]
        hdi = az.hdi(delta, hdi_prob=0.95)
        p_neg = float((delta < 0).mean())
        return {
            "gene": gene,
            "rho_hot_median": float(np.median(rh)),
            "rho_cold_median": float(np.median(rc)),
            "delta_median": float(np.median(delta)),
            "delta_hdi_low": float(hdi[0]),
            "delta_hdi_high": float(hdi[1]),
            "prob_delta_negative": p_neg,
            "posterior_significant": bool(hdi[1] < 0),  # 95% HDI excludes 0 (hot-specific)
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("posterior-diff failed for %s: %s", gene, exc)
        return None


def main():
    out = ensure_dir(Path(CFG["output_dir"]) / "immune_context_solid")
    log.info("Loading DepMap data")
    expr = load_expression()
    crispr = load_crispr()
    info = load_cell_line_info()
    lcol = "OncotreeLineage" if "OncotreeLineage" in info.columns else "PrimaryDisease"
    solid = info.index[~info[lcol].isin(HEME)]
    expr_s = expr.loc[expr.index.intersection(solid)]
    dep_s = crispr.loc[crispr.index.intersection(solid)]

    status = stratify_immune(run_estimate(expr_s), method="median")
    common = expr_s.index.intersection(dep_s.index)
    ids_hot = common.intersection(status[status == "immune_hot"].index)
    ids_cold = common.intersection(status[status == "immune_cold"].index)
    log.info("Solid paired lines: hot=%d cold=%d", len(ids_hot), len(ids_cold))

    # Candidate genes: those the Fisher-z solid analysis flagged FDR<0.05
    fisher = pd.read_csv(out / "differential_edd_solid_hot_vs_cold.csv")
    cand = fisher[fisher["fdr"] < 0.05]["gene"].tolist()
    log.info("Testing %d Fisher-z FDR<0.05 genes with posterior-difference", len(cand))

    gene_data = []
    for g in cand:
        if g in expr_s.columns and g in dep_s.columns:
            gene_data.append((
                expr_s.loc[ids_hot, g].values, dep_s.loc[ids_hot, g].values,
                expr_s.loc[ids_cold, g].values, dep_s.loc[ids_cold, g].values, g,
            ))

    results = Parallel(n_jobs=12, backend="loky", verbose=5)(
        delayed(_posterior_diff_gene)(*gd) for gd in gene_data
    )
    df = pd.DataFrame([r for r in results if r is not None])

    # Merge Fisher-z verdict for head-to-head comparison
    fz = fisher[["gene", "delta_rho", "pvalue", "fdr"]].rename(
        columns={"delta_rho": "fisher_delta", "pvalue": "fisher_p", "fdr": "fisher_fdr"})
    df = df.merge(fz, on="gene", how="left")
    df["fisher_hot_significant"] = (df["fisher_fdr"] < 0.05) & (df["fisher_delta"] < 0)
    df = df.sort_values("prob_delta_negative", ascending=False)
    df.to_csv(out / "posterior_difference_solid.csv", index=False)

    n_post = int(df["posterior_significant"].sum())
    n_fisher = int(df["fisher_hot_significant"].sum())
    agree = int((df["posterior_significant"] & df["fisher_hot_significant"]).sum())
    log.info("Posterior-significant (95%% HDI of delta excludes 0): %d", n_post)
    log.info("Fisher-z hot-significant (FDR<0.05 & delta<0): %d", n_fisher)
    log.info("Concordant (both): %d", agree)
    log.info("Top hot-specific by P(delta<0):\n%s",
             df.head(15)[["gene", "delta_median", "delta_hdi_low", "delta_hdi_high",
                          "prob_delta_negative", "posterior_significant",
                          "fisher_hot_significant"]].to_string(index=False))


if __name__ == "__main__":
    main()
