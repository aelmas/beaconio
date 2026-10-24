#!/usr/bin/env python3
"""Step 05c: ICB benchmark using SOLID-tumour-only BEACON-IO signature.

Re-derives the top-50 signature from the solid-tumour differential EDD
results (03b), then scores it on the Hugo 2016 melanoma cohort against
established immune biomarkers.

Writes to analysis/out/clinical/icb_biomarker_benchmark_solid.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from clinical.validation import benchmark_biomarkers, build_beacon_io_signature
from data.icb_cohorts import load_hugo_2016

log = get_logger("05c_icb_solid")


def main():
    out = ensure_dir(Path(CFG["output_dir"]) / "clinical")
    solid_dir = Path(CFG["output_dir"]) / "immune_context_solid"
    diff_path = solid_dir / "differential_edd_solid_hot_vs_cold.csv"

    if not diff_path.exists():
        log.error("Run 03b_immune_context_solid.py first")
        return

    diff_edd = pd.read_csv(diff_path)
    signature_genes = build_beacon_io_signature(diff_edd, top_n=50)
    log.info("Solid-tumour BEACON-IO signature: %d genes", len(signature_genes))
    log.info("First 20 genes: %s", signature_genes[:20])

    cohort = load_hugo_2016(Path(CFG["data_dir"]))
    expr = cohort["expression"]
    clin = cohort["clinical"]
    response = clin["response_binary"]
    log.info("Hugo_2016: n=%d (R=%d, NR=%d)",
             len(response), int(response.sum()),
             int(len(response) - response.sum()))

    bench = benchmark_biomarkers(expr, clin, response, signature_genes)
    bench["cohort"] = "Hugo_2016"
    bench["signature"] = "solid_tumor_only"
    bench.to_csv(out / "icb_biomarker_benchmark_solid.csv", index=False)
    log.info("Solid-tumour benchmark:\n%s", bench.to_string(index=False))


if __name__ == "__main__":
    main()
