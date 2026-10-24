#!/usr/bin/env python3
"""Step 01: Download all public datasets.

Downloads:
  - DepMap (CRISPR, expression, proteomics)
  - PRISM drug sensitivity
  - TCGA (expression, clinical, survival, mutations)
  - ICB cohorts (GEO, cBioPortal)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import get_logger
from data.depmap import download_depmap, download_prism
from data.tcga import download_tcga

log = get_logger("01_download")


def main():
    parser = argparse.ArgumentParser(description="Download all BEACON-IO datasets")
    parser.add_argument("--data-dir", default=CFG["data_dir"])
    parser.add_argument("--skip-depmap", action="store_true")
    parser.add_argument("--skip-prism", action="store_true")
    parser.add_argument("--skip-tcga", action="store_true")
    parser.add_argument("--skip-icb", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not args.skip_depmap:
        log.info("=== Downloading DepMap ===")
        download_depmap(data_dir)

    if not args.skip_prism:
        log.info("=== Downloading PRISM ===")
        download_prism(data_dir)

    if not args.skip_tcga:
        log.info("=== Downloading TCGA ===")
        download_tcga(data_dir)

    if not args.skip_icb:
        log.info("=== Downloading ICB cohorts ===")
        from data.icb_cohorts import load_all_icb_cohorts
        load_all_icb_cohorts(data_dir)

    log.info("All downloads complete.")


if __name__ == "__main__":
    main()
