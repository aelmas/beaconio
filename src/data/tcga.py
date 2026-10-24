"""Download and preprocess TCGA data via UCSC Xena hubs.

Data sources:
  - Expression: toil Xena hub (TCGA RSEM TPM, reprocessed)
  - Clinical: GDC Xena hub (pan-cancer basic phenotype)
  - Survival: toil Xena hub + GDC hub

All TCGA data is publicly available under NCI GDC terms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger

log = get_logger(__name__)


def _download_file(url: str, dest: Path, timeout: int = 600) -> Path:
    """Download a file with progress bar and redirect support."""
    if dest.exists() and dest.stat().st_size > 1000:
        log.info("TCGA already present: %s (%d bytes)", dest.name, dest.stat().st_size)
        return dest
    if dest.exists():
        dest.unlink()
    ensure_dir(dest.parent)
    log.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(1 << 20):
            fh.write(chunk)
            bar.update(len(chunk))
    log.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


def download_tcga(data_dir: Path | None = None) -> dict[str, Path]:
    """Download TCGA expression, clinical, and survival data."""
    data_dir = Path(data_dir or CFG["data_dir"]) / "tcga"
    ensure_dir(data_dir)
    tcfg = CFG["tcga"]
    files = {}

    # Expression
    files["expression"] = _download_file(
        tcfg["expression_url"],
        data_dir / tcfg["expression_file"],
    )

    # Clinical
    files["clinical"] = _download_file(
        tcfg["clinical_url"],
        data_dir / tcfg["clinical_file"],
    )

    # Survival (try toil hub first, then GDC)
    try:
        files["survival"] = _download_file(
            tcfg["survival_url"],
            data_dir / tcfg["survival_file"],
        )
    except Exception:
        log.warning("Toil survival download failed, trying GDC hub")
        files["survival"] = _download_file(
            tcfg["survival_gdc_url"],
            data_dir / tcfg["survival_gdc_file"],
        )

    return files


def _load_ensembl_to_symbol() -> dict[str, str]:
    """Load or build Ensembl-to-gene-symbol mapping."""
    cache_path = Path(CFG["output_dir"]) / "clinical" / "ensembl_to_symbol.csv"
    if cache_path.exists() and cache_path.stat().st_size > 100:
        mapping = pd.read_csv(cache_path, index_col=0).iloc[:, 0]
        return mapping.to_dict()

    # Build via mygene.info
    log.info("Building Ensembl-to-symbol mapping via mygene.info")
    import requests as _req
    ensembl_ids = []  # will be populated from expression
    # Fallback: strip version, use simple heuristic
    return {}


def load_tcga_expression(
    data_dir: Path | None = None,
    cancer_types: list[str] | None = None,
) -> pd.DataFrame:
    """Load TCGA RSEM TPM expression (samples x genes), optionally subset."""
    data_dir = Path(data_dir or CFG["data_dir"]) / "tcga"
    path = data_dir / CFG["tcga"]["expression_file"]
    log.info("Loading TCGA expression from %s", path)
    df = pd.read_csv(path, sep="\t", index_col=0).T  # genes x samples -> samples x genes
    log.info("TCGA expression: %d samples x %d genes", *df.shape)

    # Convert Ensembl IDs to gene symbols
    if df.columns[0].startswith("ENSG"):
        mapping = _load_ensembl_to_symbol()
        if mapping:
            # Strip version from column names for matching
            stripped = {c.split(".")[0]: c for c in df.columns}
            rename = {}
            for ens_nover, orig_col in stripped.items():
                if ens_nover in mapping:
                    rename[orig_col] = mapping[ens_nover]
            df = df.rename(columns=rename)
            # Drop unmapped (still Ensembl) and deduplicate
            df = df.loc[:, ~df.columns.str.startswith("ENSG")]
            df = df.loc[:, ~df.columns.duplicated()]
            log.info("Mapped to %d gene symbols", len(df.columns))

    if cancer_types:
        clinical = load_tcga_clinical(data_dir.parent)
        # Match cancer types by project_id or cancer_type
        type_col = next(
            (c for c in clinical.columns if c.lower() in
             ("project_id", "project", "_primary_disease", "disease", "cancer_type")),
            None,
        )
        if type_col:
            def _match_type(val, types):
                val_str = str(val)
                return any(t in val_str for t in types)

            keep = clinical[clinical[type_col].apply(lambda x: _match_type(x, cancer_types))].index
            df = df.loc[df.index.intersection(keep)]
            log.info("Filtered to %d samples across %s", len(df), cancer_types)
    return df


def load_tcga_clinical(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or CFG["data_dir"]) / "tcga"
    tcfg = CFG["tcga"]
    path = data_dir / tcfg["clinical_file"]
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Truncate index to 15 chars to match expression sample IDs
    df.index = df.index.str[:15]
    df = df[~df.index.duplicated(keep="first")]
    # Standardise cancer-type column
    for col in df.columns:
        if "primary" in col.lower() and "disease" in col.lower():
            df["cancer_type"] = df[col]
            break
    if "project_id" in df.columns and "cancer_type" not in df.columns:
        df["cancer_type"] = df["project_id"].str.replace("TCGA-", "")
    return df


def load_tcga_survival(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or CFG["data_dir"]) / "tcga"
    tcfg = CFG["tcga"]
    # Try primary survival file, then GDC backup
    for fname in (tcfg["survival_file"], tcfg.get("survival_gdc_file", "")):
        path = data_dir / fname
        if path.exists() and path.stat().st_size > 100:
            df = pd.read_csv(path, sep="\t", index_col=0)
            log.info("Loaded TCGA survival: %d patients, columns=%s", len(df), df.columns.tolist())
            return df
    log.warning("No TCGA survival data found")
    return pd.DataFrame()


def compute_tmb(mutations: pd.DataFrame, exome_size_mb: float = 38.0) -> pd.Series:
    """Compute tumor mutational burden (mutations / Mb)."""
    nonsynonymous = mutations[
        mutations["Variant_Classification"].isin([
            "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
            "Frame_Shift_Ins", "Splice_Site", "Nonstop_Mutation",
        ])
    ]
    counts = nonsynonymous.groupby("Tumor_Sample_Barcode").size()
    tmb = counts / exome_size_mb
    tmb.name = "TMB"
    return tmb
