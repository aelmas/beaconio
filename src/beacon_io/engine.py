"""BEACON engine: Bayesian Estimation of Correlation coefficients.

Implements the core BEACON method from Elmas et al. (GigaScience 2026) for
identifying expression-driven dependencies (EDD).  Uses PyMC for MCMC
sampling of bivariate normal posterior to estimate rho between gene
expression and CRISPR dependency.

Extended here with:
  - Immune-context stratification (TME-high vs TME-low)
  - Protein-level concordance scoring
  - Parallelised gene-wise inference
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

from beacon_io.config import CFG
from beacon_io.utils import fdr_correction, get_logger

log = get_logger(__name__)

BEACON_CFG = CFG["beacon"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BeaconResult:
    """Container for per-gene BEACON inference results."""

    gene: str
    lineage: str
    rho_posterior_median: float
    rho_hdi_low: float
    rho_hdi_high: float
    prob_negative: float          # P(rho < 0 | data)
    prob_outside_rope: float      # P(|rho| > ROPE | data)
    n_samples: int
    ess_bulk: float               # effective sample size
    rhat: float                   # convergence diagnostic
    significant: bool = False
    fdr: float = 1.0


@dataclass
class BeaconSummary:
    """Aggregated BEACON results for a lineage or pan-cancer run."""

    lineage: str
    results: list[BeaconResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])

    @property
    def significant_genes(self) -> list[str]:
        return [r.gene for r in self.results if r.significant]


# ---------------------------------------------------------------------------
# Core Bayesian model
# ---------------------------------------------------------------------------

def _beacon_single_gene(
    expression: np.ndarray,
    dependency: np.ndarray,
    gene: str,
    lineage: str,
) -> BeaconResult:
    """Run BEACON for a single gene: estimate rho via MCMC.

    Model:
        (expr_i, dep_i) ~ BivariateNormal(mu, Sigma)
        rho ~ Uniform(-1, 1)         [non-informative prior]
        sigma_expr ~ HalfCauchy(2.5)
        sigma_dep  ~ HalfCauchy(2.5)
        mu_expr ~ Normal(0, 10)
        mu_dep  ~ Normal(0, 10)
    """
    n = len(expression)

    # Standardise for numerical stability
    expr_z = (expression - expression.mean()) / (expression.std() + 1e-8)
    dep_z = (dependency - dependency.mean()) / (dependency.std() + 1e-8)
    obs = np.column_stack([expr_z, dep_z])

    with pm.Model() as model:
        # Priors
        mu = pm.Normal("mu", mu=0, sigma=2, shape=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5, shape=2)
        rho = pm.Uniform("rho", lower=-1, upper=1)

        # Construct covariance
        cov = pm.math.stack([
            [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1] ** 2],
        ])

        # Likelihood
        pm.MvNormal("obs", mu=mu, cov=cov, observed=obs)

        # Sample (cores=1 to avoid conflict with joblib gene-level parallelism)
        trace = pm.sample(
            draws=BEACON_CFG["n_draws"],
            tune=BEACON_CFG["n_tune"],
            chains=BEACON_CFG["n_chains"],
            cores=1,
            target_accept=BEACON_CFG["target_accept"],
            return_inferencedata=True,
            progressbar=False,
            random_seed=42,
        )

    # Extract posterior for rho
    rho_post = trace.posterior["rho"].values.flatten()
    hdi = az.hdi(trace, var_names=["rho"], hdi_prob=BEACON_CFG["hdi_prob"])
    rho_hdi = hdi["rho"].values

    # Diagnostics
    summary = az.summary(trace, var_names=["rho"])
    ess = summary["ess_bulk"].values[0]
    rhat = summary["r_hat"].values[0]

    # Probabilities
    prob_neg = (rho_post < 0).mean()
    rope = BEACON_CFG["rope_width"]
    prob_outside = (np.abs(rho_post) > rope).mean()

    return BeaconResult(
        gene=gene,
        lineage=lineage,
        rho_posterior_median=float(np.median(rho_post)),
        rho_hdi_low=float(rho_hdi[0]),
        rho_hdi_high=float(rho_hdi[1]),
        prob_negative=float(prob_neg),
        prob_outside_rope=float(prob_outside),
        n_samples=n,
        ess_bulk=float(ess),
        rhat=float(rhat),
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _run_single_gene_safe(
    expr_vals: np.ndarray,
    dep_vals: np.ndarray,
    gene: str,
    lineage: str,
) -> BeaconResult | None:
    """Wrapper for parallel execution — catches exceptions."""
    if expr_vals.std() < 1e-8 or dep_vals.std() < 1e-8:
        return None
    try:
        return _beacon_single_gene(expr_vals, dep_vals, gene, lineage)
    except Exception as exc:
        log.warning("BEACON failed for %s/%s: %s", lineage, gene, exc)
        return None


def run_beacon_lineage(
    expression: pd.DataFrame,
    dependency: pd.DataFrame,
    lineage: str,
    genes: list[str] | None = None,
    n_jobs: int = -1,
) -> BeaconSummary:
    """Run BEACON across all genes for a single lineage.

    Parameters
    ----------
    expression : DataFrame (cell_lines x genes), already subset to lineage.
    dependency : DataFrame (cell_lines x genes), already subset to lineage.
    lineage    : Lineage label.
    genes      : Optional subset of genes to test.
    n_jobs     : Number of parallel jobs (-1 = all CPUs).
    """
    from joblib import Parallel, delayed
    from beacon_io.utils import align_matrices

    expr, dep = align_matrices(expression, dependency)
    if genes:
        shared = [g for g in genes if g in expr.columns]
        expr, dep = expr[shared], dep[shared]

    n_lines = len(expr)
    if n_lines < BEACON_CFG["min_cell_lines"]:
        log.warning(
            "Lineage %s has only %d cell lines (min=%d), skipping",
            lineage, n_lines, BEACON_CFG["min_cell_lines"],
        )
        return BeaconSummary(lineage=lineage)

    log.info("Running BEACON MCMC on %s: %d cell lines, %d genes (n_jobs=%s)",
             lineage, n_lines, len(expr.columns), n_jobs)

    # Pre-extract numpy arrays for parallel execution
    gene_data = [
        (expr[gene].values, dep[gene].values, gene)
        for gene in expr.columns
    ]

    raw_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_run_single_gene_safe)(e, d, g, lineage)
        for e, d, g in gene_data
    )
    results = [r for r in raw_results if r is not None]

    # FDR correction on prob_outside_rope (1 - prob as p-value proxy)
    if results:
        pvals = np.array([1.0 - r.prob_outside_rope for r in results])
        fdrs = fdr_correction(pvals)
        rho_thresh = BEACON_CFG["rho_threshold"]
        fdr_thresh = BEACON_CFG["fdr_threshold"]
        for r, fdr_val in zip(results, fdrs):
            r.fdr = float(fdr_val)
            r.significant = (
                r.rho_posterior_median < rho_thresh
                and fdr_val < fdr_thresh
            )

    summary = BeaconSummary(lineage=lineage, results=results)
    log.info(
        "Lineage %s: %d significant EDD genes (of %d tested)",
        lineage, len(summary.significant_genes), len(results),
    )
    return summary


def run_beacon_pan_lineage(
    expression: pd.DataFrame,
    dependency: pd.DataFrame,
    cell_info: pd.DataFrame,
    lineage_col: str = "OncotreeLineage",
    genes: list[str] | None = None,
    n_jobs: int = -1,
) -> dict[str, BeaconSummary]:
    """Run BEACON per-lineage across all lineages in cell_info."""
    results = {}
    for lineage, group in cell_info.groupby(lineage_col):
        cell_ids = group.index.intersection(expression.index).intersection(dependency.index)
        if len(cell_ids) < BEACON_CFG["min_cell_lines"]:
            continue
        summary = run_beacon_lineage(
            expression.loc[cell_ids],
            dependency.loc[cell_ids],
            lineage=str(lineage),
            genes=genes,
            n_jobs=n_jobs,
        )
        results[str(lineage)] = summary
    return results


# ---------------------------------------------------------------------------
# Fast frequentist fallback (for screening / large gene sets)
# ---------------------------------------------------------------------------

def beacon_fast(
    expression: pd.DataFrame,
    dependency: pd.DataFrame,
) -> pd.DataFrame:
    """Fast Spearman-based screening (non-Bayesian). Use for pre-filtering
    before full MCMC, or for benchmarking comparisons.

    Returns DataFrame with columns: gene, rho, pvalue, fdr.
    """
    from beacon_io.utils import align_matrices

    expr, dep = align_matrices(expression, dependency)
    records = []
    for gene in expr.columns:
        e = expr[gene].values
        d = dep[gene].values
        if e.std() < 1e-8 or d.std() < 1e-8:
            continue
        rho, pval = stats.spearmanr(e, d)
        records.append({"gene": gene, "rho": rho, "pvalue": pval})
    df = pd.DataFrame(records)
    if not df.empty:
        df["fdr"] = fdr_correction(df["pvalue"].values)
    return df
