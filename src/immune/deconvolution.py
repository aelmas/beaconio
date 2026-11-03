"""Tumor microenvironment (TME) deconvolution and immune scoring.

Estimates immune cell-type fractions from bulk RNA-seq using multiple
methods, then stratifies samples into immune-hot / immune-cold groups
for BEACON-IO analysis.

Supported methods:
  - MCPcounter (via decoupler)
  - xCell signatures
  - EPIC
  - ESTIMATE (tumour purity / immune score)
  - TIDE score (via API)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from beacon_io.config import CFG
from beacon_io.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# MCPcounter via decoupler
# ---------------------------------------------------------------------------

def run_mcpcounter(expression: pd.DataFrame) -> pd.DataFrame:
    """Estimate immune / stromal cell abundances via MCPcounter signatures.

    Parameters
    ----------
    expression : samples x genes (log2 TPM or similar)

    Returns
    -------
    scores : samples x cell_types
    """
    import decoupler as dc

    # MCPcounter gene sets ship with decoupler
    model = dc.get_resource("MCPcounter")
    # decoupler expects genes as rows
    acts, _ = dc.run_mlm(expression.T, model, source="cell_type", target="genesymbol")
    return acts.T  # samples x cell_types


# ---------------------------------------------------------------------------
# ESTIMATE (tumour purity, immune score, stromal score)
# ---------------------------------------------------------------------------

# ESTIMATE gene lists (Yoshihara et al. 2013)
_ESTIMATE_IMMUNE_GENES = [
    "ADAMDEC1", "APOBEC3G", "BATF", "BCL2A1", "BIN2", "BTK", "C1QA",
    "C1QB", "C1QC", "CCL13", "CCL18", "CCL19", "CCL21", "CCL5", "CCR7",
    "CD19", "CD1E", "CD2", "CD247", "CD27", "CD3D", "CD3E", "CD3G",
    "CD40LG", "CD48", "CD52", "CD53", "CD69", "CD72", "CD79A", "CD79B",
    "CD86", "CLEC10A", "CLIC2", "CSF2RB", "CTSS", "CXCL10", "CXCL11",
    "CXCL13", "CXCL9", "CYBB", "DOCK2", "EVI2A", "EVI2B", "FGL2",
]

_ESTIMATE_STROMAL_GENES = [
    "ADAM12", "ASPN", "BGN", "CDH11", "COL10A1", "COL11A1", "COL1A2",
    "COL3A1", "COL5A1", "COL5A2", "COL6A3", "COMP", "CTSK", "DCN",
    "DNM3OS", "ECM2", "FAP", "FBN1", "FN1", "GREM1", "INHBA", "ISLR",
    "LRRC15", "LUM", "MFAP5", "MMP2", "MMP3", "OLFML2B", "PCOLCE",
    "POSTN", "SPARC", "SULF1", "THY1", "THBS2", "TIMP3", "VCAN",
]


def run_estimate(expression: pd.DataFrame) -> pd.DataFrame:
    """Compute ESTIMATE immune, stromal, and purity scores."""
    immune_genes = [g for g in _ESTIMATE_IMMUNE_GENES if g in expression.columns]
    stromal_genes = [g for g in _ESTIMATE_STROMAL_GENES if g in expression.columns]

    immune_score = expression[immune_genes].mean(axis=1)
    stromal_score = expression[stromal_genes].mean(axis=1)
    estimate_score = immune_score + stromal_score
    # Approximate tumour purity (Carter et al. formula)
    purity = np.cos(0.6049872018 + 0.0001467884 * estimate_score)

    return pd.DataFrame({
        "ESTIMATE_ImmuneScore": immune_score,
        "ESTIMATE_StromalScore": stromal_score,
        "ESTIMATE_TumorPurity": purity,
    }, index=expression.index)


# ---------------------------------------------------------------------------
# Immune evasion scoring
# ---------------------------------------------------------------------------

# Curated gene sets for immune evasion programmes
IMMUNE_EVASION_PROGRAMS = {
    "antigen_presentation_loss": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "CALR", "CANX", "ERAP1", "ERAP2",
    ],
    "ifn_gamma_signaling": [
        "IFNG", "IFNGR1", "IFNGR2", "JAK1", "JAK2", "STAT1",
        "IRF1", "IRF9", "PSMB10",
    ],
    "tgfb_exclusion": [
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2",
        "SMAD3", "SMAD4", "ACTA2", "COL1A1", "COL3A1", "FN1",
    ],
    "wnt_beta_catenin": [
        "CTNNB1", "APC", "AXIN1", "AXIN2", "WNT1", "WNT5A",
        "TCF7L2", "LEF1", "MYC", "CCND1",
    ],
    "immune_checkpoints": [
        "CD274", "PDCD1LG2", "CTLA4", "PDCD1", "LAG3", "HAVCR2",
        "TIGIT", "BTLA", "VSIR", "IDO1", "CD47", "SIGLEC15",
    ],
    "myeloid_suppression": [
        "CD163", "MRC1", "CSF1R", "CSF1", "IL10", "TGFB1",
        "ARG1", "NOS2", "IDO1", "VEGFA", "IL6", "CCL2",
    ],
}


def score_immune_evasion(expression: pd.DataFrame) -> pd.DataFrame:
    """Score each sample for immune evasion programmes (mean z-score)."""
    from scipy.stats import zscore

    scores = {}
    for program, genes in IMMUNE_EVASION_PROGRAMS.items():
        available = [g for g in genes if g in expression.columns]
        if len(available) < 3:
            log.warning("Too few genes for %s (%d), skipping", program, len(available))
            continue
        z = pd.DataFrame(
            zscore(expression[available], axis=0, nan_policy="omit"),
            index=expression.index,
            columns=available,
        )
        scores[program] = z.mean(axis=1)
    return pd.DataFrame(scores)


# ---------------------------------------------------------------------------
# Stratification: immune-hot vs immune-cold
# ---------------------------------------------------------------------------

def stratify_immune(
    immune_scores: pd.DataFrame,
    method: str = "median",
    score_col: str = "ESTIMATE_ImmuneScore",
) -> pd.Series:
    """Split samples into immune-hot / immune-cold.

    Parameters
    ----------
    method : "median" for median split, "cluster" for hierarchical clustering.
    """
    if method == "median":
        threshold = immune_scores[score_col].median()
        labels = (immune_scores[score_col] >= threshold).map(
            {True: "immune_hot", False: "immune_cold"}
        )
    elif method == "cluster":
        Z = linkage(immune_scores.values, method="ward")
        clusters = fcluster(Z, t=2, criterion="maxclust")
        # Label the cluster with higher mean immune score as "hot"
        means = immune_scores.groupby(clusters).mean().mean(axis=1)
        hot_cluster = means.idxmax()
        labels = pd.Series(
            ["immune_hot" if c == hot_cluster else "immune_cold" for c in clusters],
            index=immune_scores.index,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    labels.name = "immune_status"
    return labels
