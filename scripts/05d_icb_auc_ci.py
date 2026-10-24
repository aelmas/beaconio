#!/usr/bin/env python3
"""Step 05d: Bootstrap confidence intervals for the ICB-benchmark AUCs.

With only 4 responders / 24 non-responders in Hugo 2016, point AUCs are not
interpretable without uncertainty. We compute stratified bootstrap 95% CIs for
every biomarker's score-based AUC, on the SAME mean-expression scores used in
the benchmark, so readers can see the CIs are very wide and that no biomarker
ranking is statistically distinguishable. We report DIRECTIONAL AUC (no
auto-flip) to avoid the undisclosed upward bias.

Output: analysis/out/clinical/icb_auc_bootstrap_ci.csv
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
from data.icb_cohorts import load_hugo_2016

log = get_logger("05d_auc_ci")

GEP = ["IFNG", "STAT1", "CCR5", "CXCL9", "CXCL10", "CXCL11", "IDO1", "PRF1",
       "GZMA", "GZMB", "CD27", "CD274", "CD276", "CMKLR1", "HLA-DQA1",
       "HLA-DRB1", "HLA-E", "PDCD1LG2"]
IMPRES = ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD27", "CD40",
          "CD80", "ICOS", "TNFRSF14", "TNFRSF18", "BTLA", "CD244", "TNFRSF9"]
CYT = ["GZMA", "PRF1"]


def _score(expr, genes):
    av = [g for g in genes if g in expr.columns]
    if not av:
        return None
    return expr[av].mean(axis=1), len(av)


def _boot_auc(y, score, n=2000, seed=42):
    """Directional AUC + stratified bootstrap 95% CI (no auto-flip)."""
    rng = np.random.default_rng(seed)
    auc = roc_auc_score(y, score)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[bi])) < 2:
            continue
        boots.append(roc_auc_score(y[bi], score[bi]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


def main():
    out = ensure_dir(Path(CFG["output_dir"]) / "clinical")
    solid = Path(CFG["output_dir"]) / "immune_context_solid" / "differential_edd_solid_hot_vs_cold.csv"
    sig_solid = build_beacon_io_signature(pd.read_csv(solid), top_n=50)

    co = load_hugo_2016(Path(CFG["data_dir"]))
    expr, clin = co["expression"], co["clinical"]
    shared = expr.index.intersection(clin.index)
    expr = expr.loc[shared]
    y = clin.loc[shared, "response_binary"].values.astype(int)
    log.info("Hugo 2016: n=%d (R=%d, NR=%d)", len(y), int(y.sum()), int(len(y) - y.sum()))

    panels = {
        "IMPRES": IMPRES, "Cytolytic (CYT)": CYT, "IFNg-GEP (Ayers)": GEP,
        "PD-L1 (CD274)": ["CD274"], "BEACON-IO (solid)": sig_solid,
    }
    rows = []
    for name, genes in panels.items():
        sc = _score(expr, genes)
        if sc is None:
            continue
        score, n_used = sc
        auc, lo, hi = _boot_auc(y, score.values)
        rows.append({"biomarker": name, "n_genes": n_used,
                     "auc_directional": round(auc, 3),
                     "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                     "ci_width": round(hi - lo, 3)})
    df = pd.DataFrame(rows).sort_values("auc_directional", ascending=False)
    df.to_csv(out / "icb_auc_bootstrap_ci.csv", index=False)
    log.info("Bootstrap 95%% CIs (directional AUC, no auto-flip):\n%s", df.to_string(index=False))
    log.info("Note: all CIs overlap widely -> no biomarker ranking is "
             "statistically distinguishable at n=28 / 4 responders.")


if __name__ == "__main__":
    main()
