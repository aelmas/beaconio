"""Download and preprocess DepMap data (CRISPR, expression, proteomics, PRISM).

Data source: https://depmap.org/portal/download/
24Q4 hosted on Figshare: https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248
Licence: CC BY 4.0 (Achilles), custom ToS (CCLE/DepMap).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger

log = get_logger(__name__)

# DepMap 24Q4 on Figshare — individual file direct-download URLs
DEPMAP_24Q4_URLS = {
    "CRISPRGeneEffect.csv": "https://ndownloader.figshare.com/files/51064667",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv": "https://ndownloader.figshare.com/files/51065489",
    "Model.csv": "https://ndownloader.figshare.com/files/51065297",
    "OmicsCNGene.csv": "https://ndownloader.figshare.com/files/51065324",
    "OmicsSomaticMutations.csv": "https://ndownloader.figshare.com/files/51065732",
}

# PRISM Repurposing 24Q2 on Figshare
PRISM_24Q2_URLS = {
    "PRISM_Repurposing_24Q2_LFC.csv": "https://ndownloader.figshare.com/files/46630987",
    "PRISM_Repurposing_24Q2_Treatment_Info.csv": "https://ndownloader.figshare.com/files/46631146",
    "PRISM_Repurposing_24Q2_Cell_Line_Info.csv": "https://ndownloader.figshare.com/files/46630978",
}

# Figshare API for dynamic URL resolution (fallback)
FIGSHARE_ARTICLE_24Q4 = "https://api.figshare.com/v2/articles/27993248"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, chunk_size: int = 1 << 20, timeout: int = 600) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        log.info("Already downloaded: %s (%d bytes)", dest.name, dest.stat().st_size)
        return dest
    # Remove any previously downloaded bad file
    if dest.exists():
        dest.unlink()
    ensure_dir(dest.parent)
    log.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))
    log.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


def _resolve_figshare_urls(article_id: str = "27993248") -> dict[str, str]:
    """Resolve file download URLs from Figshare article API (fallback)."""
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {f["name"]: f["download_url"] for f in data.get("files", [])}


def download_depmap(data_dir: Path | None = None) -> dict[str, Path]:
    """Download core DepMap 24Q4 matrices from Figshare."""
    data_dir = Path(data_dir or CFG["data_dir"])

    # Use hardcoded URLs first, fall back to API resolution
    url_map = dict(DEPMAP_24Q4_URLS)
    files = {}
    for key in ("crispr_file", "expression_file", "cell_line_info",
                "copy_number_file", "mutation_file"):
        fname = CFG["depmap"][key]
        if fname in url_map:
            url = url_map[fname]
        else:
            # Try Figshare API
            log.info("Resolving URL for %s via Figshare API", fname)
            resolved = _resolve_figshare_urls()
            url = resolved.get(fname)
            if not url:
                log.warning("Could not find download URL for %s, skipping", fname)
                continue
        files[key] = _download(url, data_dir / "depmap" / fname)
    return files


def download_prism(data_dir: Path | None = None) -> dict[str, Path]:
    """Download PRISM Repurposing 24Q2 data from Figshare."""
    data_dir = Path(data_dir or CFG["data_dir"])
    files = {}
    for local_name, url in PRISM_24Q2_URLS.items():
        files[local_name] = _download(url, data_dir / "prism" / local_name)
    return files


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_cell_line_info(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or CFG["data_dir"])
    path = data_dir / "depmap" / CFG["depmap"]["cell_line_info"]
    df = pd.read_csv(path, index_col=0)
    if "OncotreeLineage" in df.columns:
        df["PrimaryDisease"] = df["OncotreeLineage"]
    return df


def load_crispr(data_dir: Path | None = None) -> pd.DataFrame:
    """Load Chronos CRISPR dependency scores (cell-lines x genes)."""
    data_dir = Path(data_dir or CFG["data_dir"])
    path = data_dir / "depmap" / CFG["depmap"]["crispr_file"]
    df = pd.read_csv(path, index_col=0)
    # Strip gene-ID suffix: "ACE2 (59272)" -> "ACE2"
    df.columns = [c.split(" (")[0] for c in df.columns]
    log.info("CRISPR matrix: %d cell lines x %d genes", *df.shape)
    return df


def load_expression(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or CFG["data_dir"])
    path = data_dir / "depmap" / CFG["depmap"]["expression_file"]
    df = pd.read_csv(path, index_col=0)
    df.columns = [c.split(" (")[0] for c in df.columns]
    log.info("Expression matrix: %d cell lines x %d genes", *df.shape)
    return df


def load_proteomics(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or CFG["data_dir"])
    path = data_dir / "depmap" / CFG["depmap"]["proteomics_file"]
    if not path.exists():
        log.warning("Proteomics file not found: %s (download manually from Gygi lab)", path)
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.columns = [c.split(" (")[0] for c in df.columns]
    log.info("Proteomics matrix: %d cell lines x %d proteins", *df.shape)
    return df


def load_prism(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sensitivity_matrix, treatment_info).

    PRISM 24Q2 is in long format: row_id, profile_id, LFC, LFC_cb, PASS.
    row_id = "ACH-XXXXXX::plate::screen::replicate"
    We pivot to cell_line x compound (broad_id) matrix, averaging replicates.
    """
    import numpy as np

    data_dir = Path(data_dir or CFG["data_dir"]) / "prism"
    lfc_path = data_dir / "PRISM_Repurposing_24Q2_LFC.csv"
    info_path = data_dir / "PRISM_Repurposing_24Q2_Treatment_Info.csv"

    if not lfc_path.exists():
        log.warning("PRISM LFC file not found: %s", lfc_path)
        return pd.DataFrame(), pd.DataFrame()

    log.info("Loading PRISM 24Q2 LFC data (long format)...")
    lfc = pd.read_csv(lfc_path)
    info = pd.read_csv(info_path) if info_path.exists() else pd.DataFrame()

    # Parse cell line ID from row_id
    lfc["cell_line"] = lfc["row_id"].str.split("::").str[0]

    # Merge with treatment info to get compound names
    if not info.empty and "profile_id" in info.columns:
        # Keep unique compound info per profile
        drug_map = info[["profile_id", "broad_id", "name"]].drop_duplicates(subset=["profile_id"])
        lfc = lfc.merge(drug_map, on="profile_id", how="left")
        compound_col = "broad_id"
    else:
        lfc["broad_id"] = lfc["profile_id"]
        compound_col = "broad_id"

    # Filter to passing QC
    if "PASS" in lfc.columns:
        lfc = lfc[lfc["PASS"] == True]

    # Pivot: average LFC across replicates -> cell_line x compound
    log.info("Pivoting PRISM to cell_line x compound matrix...")
    sens = lfc.groupby(["cell_line", compound_col])["LFC"].mean().reset_index()
    sens_matrix = sens.pivot(index="cell_line", columns=compound_col, values="LFC")
    log.info("PRISM: %d cell lines x %d compounds", *sens_matrix.shape)

    return sens_matrix, info
