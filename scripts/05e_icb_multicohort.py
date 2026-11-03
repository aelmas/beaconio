#!/usr/bin/env python3
"""Step 05e: Multi-cohort ICB benchmark and meta-analysis (Hugo 2016 + Riaz 2017).

Extends the single-cohort benchmark (05c/05d) to a second independent
melanoma anti-PD1 cohort and combines them by fixed-effects
inverse-variance meta-analysis.

Three deliberate choices, each correcting a weakness of the single-cohort
analysis:

1. **Response labels come from the GEO series matrix**, never from column
   order. The previous loader hardcoded the first four samples as
   responders, which mislabelled 13 of 28 Hugo patients.
2. **Directional AUC only** -- no ``1-AUC`` auto-flip. A signature that
   anti-correlates with response scores below 0.5 and is reported that way.
3. **A random-gene background.** With n in the tens, a point AUC means
   little on its own, so each panel is compared against AUCs from
   size-matched random gene sets drawn from the same expression matrix.

Outputs (analysis/out/clinical/):
  icb_multicohort_auc.csv       per-cohort directional AUC + bootstrap CI
  icb_multicohort_meta.csv      fixed-effects meta-analysis across cohorts
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from clinical.validation import build_beacon_io_signature
from data.icb_cohorts import load_hugo_2016, load_riaz_2017

log = get_logger("05e_icb_multicohort")

GEP = ["IFNG", "STAT1", "CCR5", "CXCL9", "CXCL10", "CXCL11", "IDO1", "PRF1",
       "GZMA", "GZMB", "CD27", "CD274", "CD276", "CMKLR1", "HLA-DQA1",
       "HLA-DRB1", "HLA-E", "PDCD1LG2"]
IMPRES = ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD27", "CD40",
          "CD80", "ICOS", "TNFRSF14", "TNFRSF18", "BTLA", "CD244", "TNFRSF9"]
CYT = ["GZMA", "PRF1"]

N_BOOT = 2000
N_RANDOM = 1000
SEED = 42


def _score(expr: pd.DataFrame, genes: list[str]) -> pd.Series | None:
    """Mean log1p expression of the available panel genes.

    log1p keeps FPKM-scale outliers from dominating the mean, and is applied
    identically to every panel so the comparison stays apples-to-apples.
    """
    available = [g for g in genes if g in expr.columns]
    if not available:
        return None
    return np.log1p(expr[available]).mean(axis=1)


def _hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUC (Hanley & McNeil 1982)."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 1e-12)))


def _boot_ci(y: np.ndarray, score: np.ndarray, n_boot: int = N_BOOT,
             seed: int = SEED) -> tuple[float, float]:
    """Stratified bootstrap 95% CI for the directional AUC."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], score[idx]))
    return tuple(np.percentile(boots, [2.5, 97.5]))


def _random_gene_null(expr: pd.DataFrame, y: np.ndarray, size: int,
                      n_random: int = N_RANDOM, seed: int = SEED) -> np.ndarray:
    """Directional AUCs from size-matched random gene sets."""
    rng = np.random.default_rng(seed)
    # Restrict to expressed genes so the null is not dominated by all-zero rows
    expressed = expr.columns[(expr > 0).mean(axis=0) > 0.25]
    logged = np.log1p(expr[expressed])
    aucs = np.empty(n_random)
    for i in range(n_random):
        pick = rng.choice(len(expressed), size=min(size, len(expressed)),
                          replace=False)
        aucs[i] = roc_auc_score(y, logged.iloc[:, pick].mean(axis=1).values)
    return aucs


def evaluate_cohort(name: str, cohort: dict, panels: dict) -> pd.DataFrame:
    expr, clin = cohort["expression"], cohort["clinical"]
    if expr.empty:
        log.warning("%s: no expression data, skipping", name)
        return pd.DataFrame()

    y = clin.loc[expr.index, "response_binary"].values.astype(int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    log.info("%s: n=%d (R=%d, NR=%d)", name, len(y), n_pos, n_neg)
    if n_pos < 3 or n_neg < 3:
        log.warning("%s: too few events, skipping", name)
        return pd.DataFrame()

    rows = []
    for panel, genes in panels.items():
        score = _score(expr, genes)
        if score is None:
            log.warning("%s / %s: no panel genes present", name, panel)
            continue
        n_used = int(score.notna().sum() and
                     len([g for g in genes if g in expr.columns]))
        auc = roc_auc_score(y, score.values)
        lo, hi = _boot_ci(y, score.values)
        null = _random_gene_null(expr, y, n_used)
        # Two-sided empirical p: how extreme is |AUC-0.5| against random sets?
        p_emp = float((np.abs(null - 0.5) >= abs(auc - 0.5)).mean())
        rows.append({
            "cohort": name,
            "biomarker": panel,
            "n_samples": len(y),
            "n_responders": n_pos,
            "n_nonresponders": n_neg,
            "n_genes": n_used,
            "auc_directional": round(auc, 3),
            "ci95_low": round(lo, 3),
            "ci95_high": round(hi, 3),
            "se_hanley": round(_hanley_mcneil_se(auc, n_pos, n_neg), 3),
            "null_mean": round(float(null.mean()), 3),
            "null_p95_abs_dev": round(float(np.percentile(np.abs(null - 0.5), 95)), 3),
            "p_empirical_vs_random": round(p_emp, 4),
        })
    return pd.DataFrame(rows)


def meta_analyse(per_cohort: pd.DataFrame) -> pd.DataFrame:
    """Fixed-effects inverse-variance meta-analysis of directional AUCs."""
    rows = []
    for panel, grp in per_cohort.groupby("biomarker"):
        aucs = grp["auc_directional"].values.astype(float)
        ses = grp["se_hanley"].values.astype(float)
        w = 1.0 / ses**2
        meta_auc = float((w * aucs).sum() / w.sum())
        se_meta = float(np.sqrt(1.0 / w.sum()))
        # Cochran's Q and I^2 for between-cohort heterogeneity
        q = float((w * (aucs - meta_auc) ** 2).sum())
        dof = len(aucs) - 1
        i2 = float(max(0.0, (q - dof) / q) * 100) if q > 0 and dof > 0 else 0.0
        rows.append({
            "biomarker": panel,
            "n_cohorts": len(aucs),
            "meta_auc": round(meta_auc, 3),
            "se": round(se_meta, 3),
            "ci95_low": round(meta_auc - 1.96 * se_meta, 3),
            "ci95_high": round(meta_auc + 1.96 * se_meta, 3),
            "cochran_q": round(q, 3),
            "i2_percent": round(i2, 1),
            "per_cohort_auc": "; ".join(
                f"{c}={a:.3f}" for c, a in zip(grp["cohort"], aucs)
            ),
        })
    return pd.DataFrame(rows).sort_values("meta_auc", ascending=False)


def main():
    out = ensure_dir(Path(CFG["output_dir"]) / "clinical")
    data_dir = Path(CFG["data_dir"])

    solid = (Path(CFG["output_dir"]) / "immune_context_solid"
             / "differential_edd_solid_hot_vs_cold.csv")
    if not solid.exists():
        log.error("Run 03b_immune_context_solid.py first (%s missing)", solid)
        return
    diff_edd = pd.read_csv(solid)
    sig_solid = build_beacon_io_signature(diff_edd, top_n=50)
    # Sensitivity: the uncorrected |delta_rho| ranking, which also admits
    # immune-cold genes. Reported so the effect of the correction is visible.
    sig_absdelta = build_beacon_io_signature(diff_edd, top_n=50,
                                             immune_specific_only=False)
    log.info("BEACON-IO solid signature: %d genes (immune-hot-specific)",
             len(sig_solid))
    log.info("Uncorrected |delta_rho| signature: %d genes", len(sig_absdelta))

    panels = {
        "BEACON-IO (solid)": sig_solid,
        "BEACON-IO (|drho|, uncorrected)": sig_absdelta,
        "IMPRES": IMPRES,
        "IFNg-GEP (Ayers)": GEP,
        "Cytolytic (CYT)": CYT,
        "PD-L1 (CD274)": ["CD274"],
    }

    cohorts = {
        "Hugo_2016": load_hugo_2016(data_dir),
        "Riaz_2017_Pre": load_riaz_2017(data_dir, visit="Pre"),
    }

    per_cohort = pd.concat(
        [evaluate_cohort(n, c, panels) for n, c in cohorts.items()],
        ignore_index=True,
    )
    if per_cohort.empty:
        log.error("No cohort produced results")
        return

    per_cohort.to_csv(out / "icb_multicohort_auc.csv", index=False)
    log.info("Per-cohort directional AUCs:\n%s", per_cohort.to_string(index=False))

    meta = meta_analyse(per_cohort)
    meta.to_csv(out / "icb_multicohort_meta.csv", index=False)
    log.info("Fixed-effects meta-analysis:\n%s", meta.to_string(index=False))


if __name__ == "__main__":
    main()
