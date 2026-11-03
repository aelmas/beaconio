#!/usr/bin/env python3
"""Step 03d (DIAGNOSTIC): does the stratifier matter?

Addresses the premise objection that DepMap cell lines have no immune
microenvironment, so an ESTIMATE "immune score" computed on them cannot be
measuring infiltration. On solid lines the ESTIMATE immune panel is largely
switched off (29 of 45 genes below detection in >50% of lines) and the score
is carried by five genes cancer cells express cell-autonomously.

This script re-stratifies the same solid lines on a *tumour-cell-intrinsic*
axis instead -- the Hallmark interferon-gamma response -- and asks whether
the differential-dependency hits survive the change of stratifier.

METHOD NOTE: both arms use Spearman + Fisher-z, not BEACON MCMC. That is
deliberate. Comparing an MCMC arm against a Spearman arm would confound the
stratifier with the estimator; running both by the same fast estimator makes
any difference attributable to the stratifier alone. The script first
validates Spearman against the existing MCMC output on the ESTIMATE
stratification, so the reader can judge how faithful the proxy is. Headline
numbers must still be regenerated with BEACON MCMC.

Output: analysis/out/immune_context_solid/ifn_vs_estimate_diagnostic.csv
        analysis/out/immune_context_solid/differential_edd_solid_ifn.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, fdr_correction, get_logger
from data.depmap import load_cell_line_info, load_crispr, load_expression
from immune.deconvolution import run_estimate

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = get_logger("03d_ifn_diagnostic")

HEME_LINEAGES = {"Lymphoid", "Myeloid"}
HALLMARK_SETS = ["HALLMARK_INTERFERON_GAMMA_RESPONSE",
                 "HALLMARK_INTERFERON_ALPHA_RESPONSE"]


def load_gmt(path: Path) -> dict:
    sets = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def signature_score(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Mean per-gene z-score across lines.

    Standardising each gene first stops a few highly expressed members from
    dominating the mean -- the failure mode that lets five genes carry half
    the variance of the ESTIMATE score.
    """
    available = [g for g in genes if g in expr.columns]
    sub = expr[available]
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    return z.mean(axis=1)


def spearman_diff(expr, dep, ids_a, ids_b, genes, label_a, label_b):
    """Per-group Spearman rho and Fisher-z differential test."""
    records = []
    for gene in genes:
        if gene not in expr.columns or gene not in dep.columns:
            continue
        ea, da = expr.loc[ids_a, gene].dropna(), dep.loc[ids_a, gene].dropna()
        ca = ea.index.intersection(da.index)
        eb, db = expr.loc[ids_b, gene].dropna(), dep.loc[ids_b, gene].dropna()
        cb = eb.index.intersection(db.index)
        if len(ca) < 10 or len(cb) < 10:
            continue
        ra, _ = stats.spearmanr(ea.loc[ca], da.loc[ca])
        rb, _ = stats.spearmanr(eb.loc[cb], db.loc[cb])
        if np.isnan(ra) or np.isnan(rb):
            continue
        za, zb = np.arctanh(np.clip([ra, rb], -0.999, 0.999))
        se = np.sqrt(1 / (len(ca) - 3) + 1 / (len(cb) - 3))
        z = (za - zb) / se
        records.append({
            "gene": gene,
            f"rho_{label_a}": ra,
            f"rho_{label_b}": rb,
            "delta_rho": ra - rb,
            "z_fisher": z,
            "pvalue": 2 * stats.norm.sf(abs(z)),
            "n_a": len(ca), "n_b": len(cb),
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["fdr"] = fdr_correction(df["pvalue"].values)
    return df


def main():
    out = ensure_dir(Path(CFG["output_dir"]) / "immune_context_solid")

    log.info("Loading DepMap data")
    expr_all, crispr, cell_info = load_expression(), load_crispr(), load_cell_line_info()
    lineage_col = ("OncotreeLineage" if "OncotreeLineage" in cell_info.columns
                   else "PrimaryDisease")
    solid_ids = cell_info.index[~cell_info[lineage_col].isin(HEME_LINEAGES)]
    expr = expr_all.loc[expr_all.index.intersection(solid_ids)]
    dep = crispr.loc[crispr.index.intersection(solid_ids)]
    log.info("Solid lines: expr=%d dep=%d", len(expr), len(dep))

    # --- two competing stratifiers on the same lines --------------------
    est = run_estimate(expr)["ESTIMATE_ImmuneScore"]
    hallmark = load_gmt(Path(CFG["data_dir"]) / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt")
    ifn_genes = sorted({g for s in HALLMARK_SETS for g in hallmark[s]})
    ifn = signature_score(expr, ifn_genes)
    log.info("Hallmark IFN genes used: %d of %d",
             len([g for g in ifn_genes if g in expr.columns]), len(ifn_genes))

    rho_ax, p_ax = stats.spearmanr(est.loc[ifn.index], ifn)
    log.info("ESTIMATE-immune vs Hallmark-IFN across solid lines: rho=%.3f (p=%.2e)",
             rho_ax, p_ax)

    strat = pd.DataFrame({"estimate": est, "ifn": ifn}).dropna()
    est_hot = strat.index[strat["estimate"] >= strat["estimate"].median()]
    est_cold = strat.index[strat["estimate"] < strat["estimate"].median()]
    ifn_hi = strat.index[strat["ifn"] >= strat["ifn"].median()]
    ifn_lo = strat.index[strat["ifn"] < strat["ifn"].median()]
    concord = len(set(est_hot) & set(ifn_hi)) + len(set(est_cold) & set(ifn_lo))
    log.info("Median-split concordance between the two stratifiers: %d/%d (%.1f%%)",
             concord, len(strat), 100 * concord / len(strat))

    for name, ids in [("ESTIMATE-hot", est_hot), ("IFN-high", ifn_hi)]:
        top = cell_info.loc[cell_info.index.intersection(ids), lineage_col] \
                       .value_counts(normalize=True).head(5) * 100
        log.info("%s lineage composition (%%):\n%s", name, top.round(1).to_string())

    # --- candidate genes: identical to 03b ------------------------------
    beacon_dir = Path(CFG["output_dir"]) / "beacon_edd"
    fast_path = beacon_dir / "fast_screen_mrna.csv"
    if fast_path.exists():
        candidates = pd.read_csv(fast_path).query("rho < -0.15")["gene"].tolist()
    else:
        candidates = pd.read_csv(beacon_dir / "beacon_edd_significant.csv")["gene"] \
                       .unique().tolist()
    log.info("Candidate genes: %d", len(candidates))

    common = expr.index.intersection(dep.index)
    def _i(ids):
        return common.intersection(ids)

    log.info("Differential EDD, ESTIMATE stratification (Spearman)")
    d_est = spearman_diff(expr, dep, _i(est_hot), _i(est_cold), candidates,
                          "hot", "cold")
    log.info("Differential EDD, IFN stratification (Spearman)")
    d_ifn = spearman_diff(expr, dep, _i(ifn_hi), _i(ifn_lo), candidates,
                          "ifn_high", "ifn_low")
    d_ifn.to_csv(out / "differential_edd_solid_ifn.csv", index=False)

    # --- is Spearman a faithful proxy for the MCMC arm? -----------------
    mcmc_path = out / "differential_edd_solid_hot_vs_cold.csv"
    if mcmc_path.exists():
        mcmc = pd.read_csv(mcmc_path)[["gene", "delta_rho"]].rename(
            columns={"delta_rho": "delta_mcmc"})
        chk = mcmc.merge(d_est[["gene", "delta_rho"]], on="gene")
        r, _ = stats.spearmanr(chk["delta_mcmc"], chk["delta_rho"])
        log.info("PROXY CHECK -- Spearman vs BEACON MCMC delta_rho on the same "
                 "stratification: rho=%.3f over %d genes", r, len(chk))

    # --- does the hit list survive the change of stratifier? ------------
    m = d_est[["gene", "delta_rho", "fdr"]].rename(
            columns={"delta_rho": "delta_estimate", "fdr": "fdr_estimate"}) \
        .merge(d_ifn[["gene", "delta_rho", "fdr"]].rename(
            columns={"delta_rho": "delta_ifn", "fdr": "fdr_ifn"}), on="gene")
    r_all, p_all = stats.spearmanr(m["delta_estimate"], m["delta_ifn"])
    log.info("delta_rho rank correlation across %d shared genes: rho=%.3f (p=%.2e)",
             len(m), r_all, p_all)

    for n in (13, 25, 50):
        a = set(m.nsmallest(n, "delta_estimate")["gene"])
        b = set(m.nsmallest(n, "delta_ifn")["gene"])
        log.info("top-%d hot/high-specific hits: overlap %d/%d (%.0f%%)",
                 n, len(a & b), n, 100 * len(a & b) / n)

    m["rank_estimate"] = m["delta_estimate"].rank()
    m["rank_ifn"] = m["delta_ifn"].rank()
    top13 = m.nsmallest(13, "delta_estimate")
    log.info("Where the 13 ESTIMATE-derived hits land under IFN stratification:\n%s",
             top13[["gene", "delta_estimate", "delta_ifn",
                    "rank_ifn", "fdr_ifn"]].round(4).to_string(index=False))
    m.sort_values("delta_estimate").to_csv(
        out / "ifn_vs_estimate_diagnostic.csv", index=False)
    log.info("Wrote %s", out / "ifn_vs_estimate_diagnostic.csv")


if __name__ == "__main__":
    main()
