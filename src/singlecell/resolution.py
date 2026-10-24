"""Single-cell resolution of BEACON-IO targets.

Uses public scRNA-seq atlases from ICB-treated tumours to:
  1. Resolve which cell type (tumour vs immune vs stroma) expresses each
     BEACON-IO target — critical for interpreting combination strategies.
  2. Identify tumour-intrinsic EDD targets that co-localise with immune
     exclusion programmes at the single-cell level.
  3. Build cell-type-specific dependency predictions using scRNA-seq
     expression + cell-line BEACON models.

Public datasets:
  - Jerby-Arnon 2018 (melanoma ICB, GSE115978)
  - Sade-Feldman 2018 (melanoma TILs, GSE120575)
  - Zhang 2021 (pan-cancer T cell atlas)
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_singlecell_atlas(
    name: str,
    data_dir: Path | None = None,
) -> ad.AnnData:
    """Load a pre-downloaded scRNA-seq atlas as AnnData.

    Expects H5AD format in data/raw/singlecell/{name}.h5ad.
    """
    data_dir = Path(data_dir or CFG["data_dir"]) / "singlecell"
    path = data_dir / f"{name}.h5ad"
    if not path.exists():
        log.warning(
            "scRNA-seq atlas %s not found at %s. "
            "Download from GEO and convert to H5AD first.",
            name, path,
        )
        return ad.AnnData()
    adata = sc.read_h5ad(path)
    log.info("Loaded %s: %d cells x %d genes", name, adata.n_obs, adata.n_vars)
    return adata


# ---------------------------------------------------------------------------
# Cell-type resolution of BEACON-IO targets
# ---------------------------------------------------------------------------

def celltype_expression_profile(
    adata: ad.AnnData,
    beacon_genes: list[str],
    celltype_col: str = "cell_type",
) -> pd.DataFrame:
    """Compute mean expression of BEACON-IO genes per cell type.

    Returns DataFrame (cell_types x genes) of mean log-normalised expression.
    """
    available = [g for g in beacon_genes if g in adata.var_names]
    if not available:
        log.warning("No BEACON-IO genes found in scRNA-seq data")
        return pd.DataFrame()

    subset = adata[:, available].copy()

    # Ensure log-normalised
    if "log1p" not in subset.uns.get("log1p", {}):
        sc.pp.normalize_total(subset, target_sum=1e4)
        sc.pp.log1p(subset)

    # Mean expression per cell type
    cell_types = subset.obs[celltype_col].unique()
    records = {}
    for ct in cell_types:
        mask = subset.obs[celltype_col] == ct
        mean_expr = np.array(subset[mask].X.mean(axis=0)).flatten()
        records[ct] = mean_expr

    return pd.DataFrame(records, index=available).T


def classify_target_compartment(
    celltype_expr: pd.DataFrame,
    tumour_types: list[str] | None = None,
    immune_types: list[str] | None = None,
) -> pd.DataFrame:
    """Classify each BEACON-IO target as tumour-intrinsic, immune-expressed,
    or stromal based on cell-type expression profiles.

    Returns DataFrame: gene, primary_compartment, tumour_fraction, immune_fraction.
    """
    if tumour_types is None:
        tumour_types = ["Malignant", "Tumor", "Cancer", "Epithelial"]
    if immune_types is None:
        immune_types = [
            "T cell", "CD8", "CD4", "NK", "B cell", "Macrophage",
            "Monocyte", "DC", "Treg", "Myeloid",
        ]

    def _match(ct, patterns):
        return any(p.lower() in ct.lower() for p in patterns)

    tumour_mask = celltype_expr.index.map(lambda x: _match(x, tumour_types))
    immune_mask = celltype_expr.index.map(lambda x: _match(x, immune_types))
    stromal_mask = ~(tumour_mask | immune_mask)

    records = []
    for gene in celltype_expr.columns:
        total = celltype_expr[gene].sum()
        if total < 1e-8:
            continue
        t_frac = celltype_expr.loc[tumour_mask, gene].sum() / total if tumour_mask.any() else 0
        i_frac = celltype_expr.loc[immune_mask, gene].sum() / total if immune_mask.any() else 0
        s_frac = celltype_expr.loc[stromal_mask, gene].sum() / total if stromal_mask.any() else 0

        if t_frac >= 0.5:
            compartment = "tumour_intrinsic"
        elif i_frac >= 0.5:
            compartment = "immune"
        elif s_frac >= 0.5:
            compartment = "stromal"
        else:
            compartment = "mixed"

        records.append({
            "gene": gene,
            "primary_compartment": compartment,
            "tumour_fraction": round(t_frac, 3),
            "immune_fraction": round(i_frac, 3),
            "stromal_fraction": round(s_frac, 3),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ICB responder vs non-responder differential expression at single-cell level
# ---------------------------------------------------------------------------

def sc_differential_beacon_targets(
    adata: ad.AnnData,
    beacon_genes: list[str],
    response_col: str = "response",
    celltype_col: str = "cell_type",
    responder_label: str = "R",
) -> pd.DataFrame:
    """Test differential expression of BEACON-IO targets between ICB
    responders and non-responders, per cell type.

    Uses Wilcoxon rank-sum test within each cell type.

    Returns DataFrame: gene, cell_type, log2fc, pvalue, fdr.
    """
    if response_col not in adata.obs.columns:
        log.warning("Response column %s not in adata.obs", response_col)
        return pd.DataFrame()

    available = [g for g in beacon_genes if g in adata.var_names]
    records = []

    for ct in adata.obs[celltype_col].unique():
        ct_data = adata[adata.obs[celltype_col] == ct]
        resp = ct_data[ct_data.obs[response_col] == responder_label]
        nonresp = ct_data[ct_data.obs[response_col] != responder_label]

        if resp.n_obs < 10 or nonresp.n_obs < 10:
            continue

        for gene in available:
            gene_idx = list(adata.var_names).index(gene)
            vals_r = np.array(resp.X[:, gene_idx].todense()).flatten() \
                if hasattr(resp.X, "todense") else resp.X[:, gene_idx].flatten()
            vals_nr = np.array(nonresp.X[:, gene_idx].todense()).flatten() \
                if hasattr(nonresp.X, "todense") else nonresp.X[:, gene_idx].flatten()

            mean_r = vals_r.mean()
            mean_nr = vals_nr.mean()
            log2fc = np.log2((mean_nr + 1e-8) / (mean_r + 1e-8))

            from scipy.stats import mannwhitneyu
            try:
                _, pval = mannwhitneyu(vals_r, vals_nr, alternative="two-sided")
            except ValueError:
                pval = 1.0

            records.append({
                "gene": gene,
                "cell_type": ct,
                "mean_responder": mean_r,
                "mean_nonresponder": mean_nr,
                "log2fc_NR_vs_R": log2fc,
                "pvalue": pval,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["fdr"] = fdr_correction(df["pvalue"].values)
    return df


# ---------------------------------------------------------------------------
# Programme scoring at single-cell level
# ---------------------------------------------------------------------------

def score_evasion_programmes_sc(
    adata: ad.AnnData,
    programmes: dict[str, list[str]] | None = None,
) -> ad.AnnData:
    """Score immune evasion programmes per cell using scanpy.tl.score_genes.

    Adds columns to adata.obs: one per programme.
    """
    from immune.deconvolution import IMMUNE_EVASION_PROGRAMS

    programmes = programmes or IMMUNE_EVASION_PROGRAMS

    for prog_name, genes in programmes.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) < 3:
            continue
        sc.tl.score_genes(adata, gene_list=available, score_name=prog_name)
        log.info("Scored %s (%d genes) across %d cells", prog_name, len(available), adata.n_obs)

    return adata
