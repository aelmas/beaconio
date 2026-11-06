#!/usr/bin/env python3
"""Step 07b: Weight-sensitivity analysis for the evidence-integration composite.

The integration composite is a weight-normalized mean of rank-normalized
tiers (E1-E4, E6 populated). Because the tier weights are a design choice,
this script quantifies how sensitive the composite ranking is to them:
for each evidence table (all-lineages and solid-tumor), it recomputes the
composite under (i) each populated tier individually halved or multiplied
by 1.5, and (ii) ten random perturbations of all populated weights by up
to +/-50%, then reports the Spearman correlation of the full ranking with
the baseline and the top-10 gene overlap.

Output:
  analysis/out/integration/weight_sensitivity.csv
      columns: table, scheme, spearman_rho, top10_overlap
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger

log = get_logger("07b_weight_sensitivity")

# Populated tiers only (E5/E7/E8 have no public data; see Methods).
BASE_WEIGHTS = {
    "E1_edd": 0.15,
    "E2_immune_specific": 0.15,
    "E3_evasion_corr": 0.10,
    "E4_prism": 0.15,
    "E6_tcga_survival": 0.10,
}

SCORE_COLS = {
    "E1_edd": ("E1_edd_rho", True),          # more negative rho = better
    "E2_immune_specific": ("E2_delta_rho", True),  # most negative = hot-specific
    "E3_evasion_corr": ("E3_evasion_rho", False),
    "E4_prism": ("E4_prism_rho", True),
    "E6_tcga_survival": ("E6_min_cox_p", True),   # smaller p = better
}


def composite(master: pd.DataFrame, weights: dict) -> np.ndarray:
    """Weight-normalized mean of rank-normalized populated tiers (as 07)."""
    ws = np.zeros(len(master))
    wp = np.zeros(len(master))
    for key, (col, smaller_is_better) in SCORE_COLS.items():
        w = weights.get(key, 0)
        if col not in master.columns:
            continue
        vals = master[col]
        present = vals.notna().values
        ranked = vals.rank(ascending=not smaller_is_better, pct=True).fillna(0.0).values
        ws += w * ranked * present
        wp += w * present
    with np.errstate(invalid="ignore", divide="ignore"):
        return ws / wp


def main() -> None:
    out_dir = ensure_dir(Path(CFG["output_dir"]) / "integration")
    rng = np.random.default_rng(42)
    rows = []
    for fname, tag in [("beacon_io_evidence_table.csv", "all_lineages"),
                       ("beacon_io_evidence_table_solid.csv", "solid")]:
        path = out_dir / fname
        if not path.exists():
            log.warning("Evidence table missing: %s", path)
            continue
        master = pd.read_csv(path)
        base = composite(master, BASE_WEIGHTS)
        base_top10 = set(
            master.assign(c=np.nan_to_num(base, nan=-1))
            .sort_values("c", ascending=False).head(10)["gene"])

        schemes: dict[str, dict] = {}
        for key, w in BASE_WEIGHTS.items():
            schemes[f"{key}_halved"] = {**BASE_WEIGHTS, key: w * 0.5}
            schemes[f"{key}_x1.5"] = {**BASE_WEIGHTS, key: w * 1.5}
        for i in range(10):
            schemes[f"random_{i + 1}"] = {
                k: w * rng.uniform(0.5, 1.5) for k, w in BASE_WEIGHTS.items()}

        for name, weights in schemes.items():
            comp = composite(master, weights)
            ok = ~np.isnan(comp) & ~np.isnan(base)
            rho = spearmanr(base[ok], comp[ok]).statistic
            top10 = set(
                master.assign(c=np.nan_to_num(comp, nan=-1))
                .sort_values("c", ascending=False).head(10)["gene"])
            rows.append({"table": tag, "scheme": name,
                         "spearman_rho": rho,
                         "top10_overlap": len(top10 & base_top10)})
        log.info("[%s] %d schemes: min rho %.4f, min top-10 overlap %d",
                 tag, len(schemes),
                 min(r["spearman_rho"] for r in rows if r["table"] == tag),
                 min(r["top10_overlap"] for r in rows if r["table"] == tag))

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "weight_sensitivity.csv", index=False)
    log.info("Saved %d rows to %s", len(out), out_dir / "weight_sensitivity.csv")


if __name__ == "__main__":
    main()
