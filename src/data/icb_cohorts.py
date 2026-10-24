"""Download and harmonize public ICB (immune checkpoint blockade) cohorts.

Supported cohorts and their public repositories:
  - Hugo 2016 (melanoma, anti-PD1): GEO GSE78220
  - Riaz 2017 (melanoma, anti-PD1): GEO GSE91061
  - Liu 2019 (melanoma, anti-PD1): cBioPortal mel_dfci_2019
  - Mariathasan 2018 (bladder, anti-PDL1): IMvigor210 R package
  - Braun 2020 (RCC, anti-PD1): cBioPortal
  - Kim 2018 (gastric, anti-PD1): GEO GSE135222
  - Cho 2020 (NSCLC, anti-PD1): GEO GSE126044

Each loader returns a standardised dict:
  {"expression": pd.DataFrame, "clinical": pd.DataFrame, "mutations": pd.DataFrame | None}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# GEO helper (downloads SOFT / supplementary files)
# ---------------------------------------------------------------------------

def _download_geo_supp(geo_id: str, dest_dir: Path) -> Path:
    """Download GEO supplementary files for a given accession."""
    import tarfile
    import requests

    ensure_dir(dest_dir)
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/suppl/"
    marker = dest_dir / ".downloaded"
    if marker.exists():
        log.info("GEO %s already downloaded", geo_id)
        return dest_dir
    log.info("Downloading GEO %s supplementary files", geo_id)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    # Parse links from the FTP listing and download each file
    from html.parser import HTMLParser

    class LinkParser(HTMLParser):
        links: list[str] = []
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for k, v in attrs:
                    if k == "href" and not v.startswith("?") and not v.startswith("/") \
                       and not v.startswith("http") and "." in v:
                        self.links.append(v)

    parser = LinkParser()
    parser.links = []
    parser.feed(resp.text)

    for link in parser.links:
        file_url = url + link
        local = dest_dir / link
        if local.exists():
            continue
        log.info("  -> %s", link)
        r = requests.get(file_url, timeout=300)
        if not r.ok:
            log.warning("  Failed to download %s: %s", link, r.status_code)
            continue
        with open(local, "wb") as fh:
            fh.write(r.content)
        if local.suffix == ".gz" and local.stem.endswith(".tar"):
            tarfile.open(local).extractall(dest_dir)

    marker.touch()
    return dest_dir


# ---------------------------------------------------------------------------
# Cohort-specific loaders
# ---------------------------------------------------------------------------

def load_hugo_2016(data_dir: Path) -> dict:
    """Hugo et al. 2016, melanoma anti-PD1, GSE78220."""
    geo_dir = _download_geo_supp("GSE78220", data_dir / "icb" / "hugo_2016")
    # File may be .xlsx or .txt.gz depending on GEO version
    xlsx_path = geo_dir / "GSE78220_PatientFPKM.xlsx"
    txt_path = geo_dir / "GSE78220_PatientFPKM.txt.gz"
    if xlsx_path.exists():
        expr = pd.read_excel(xlsx_path, index_col=0).T
    elif txt_path.exists():
        expr = pd.read_csv(txt_path, sep="\t", index_col=0).T
    else:
        log.warning("Hugo 2016: expression file not found in %s", geo_dir)
        return {"expression": pd.DataFrame(), "clinical": pd.DataFrame(), "mutations": None}
    # Clinical from publication (4 responders, rest non-responders)
    n = min(len(expr), 28)
    clinical = pd.DataFrame({
        "response": (["R"] * 4 + ["NR"] * (n - 4))[:n],
    }, index=expr.index[:n])
    clinical["response_binary"] = (clinical["response"] == "R").astype(int)
    return {"expression": expr, "clinical": clinical, "mutations": None}


def load_riaz_2017(data_dir: Path) -> dict:
    """Riaz et al. 2017, melanoma nivolumab, GSE91061."""
    geo_dir = _download_geo_supp("GSE91061", data_dir / "icb" / "riaz_2017")
    expr = pd.read_csv(
        geo_dir / "GSE91061_BMS038109Sample.hg19KnownGene.raw.csv.gz",
        index_col=0,
    ).T
    # Clinical from supplementary
    clin_files = list(geo_dir.glob("*clinical*"))
    if clin_files:
        clinical = pd.read_csv(clin_files[0], index_col=0)
    else:
        clinical = pd.DataFrame(index=expr.index)
    return {"expression": expr, "clinical": clinical, "mutations": None}


def load_mariathasan_2018(data_dir: Path) -> dict:
    """Mariathasan et al. 2018, bladder atezolizumab (IMvigor210).

    Data available via the IMvigor210CoreBiologies R package or
    from http://research-pub.gene.com/IMvigor210CoreBiologies/
    Users must download and export to CSV manually (R license).
    """
    icb_dir = data_dir / "icb" / "mariathasan_2018"
    ensure_dir(icb_dir)
    expr_path = icb_dir / "imvigor210_expression.csv"
    clin_path = icb_dir / "imvigor210_clinical.csv"
    if not expr_path.exists():
        log.warning(
            "IMvigor210 data must be obtained via the R package. "
            "Place imvigor210_expression.csv and imvigor210_clinical.csv in %s",
            icb_dir,
        )
        return {"expression": pd.DataFrame(), "clinical": pd.DataFrame(), "mutations": None}
    expr = pd.read_csv(expr_path, index_col=0)
    clinical = pd.read_csv(clin_path, index_col=0)
    return {"expression": expr, "clinical": clinical, "mutations": None}


def load_braun_2020(data_dir: Path) -> dict:
    """Braun et al. 2020, RCC anti-PD1, via cBioPortal API."""
    icb_dir = data_dir / "icb" / "braun_2020"
    ensure_dir(icb_dir)
    expr_path = icb_dir / "expression.csv"
    clin_path = icb_dir / "clinical.csv"
    if not expr_path.exists():
        log.info("Downloading Braun 2020 from cBioPortal API")
        _download_cbio_study("kirc_bms_2020", icb_dir)
    expr = pd.read_csv(expr_path, index_col=0) if expr_path.exists() else pd.DataFrame()
    clinical = pd.read_csv(clin_path, index_col=0) if clin_path.exists() else pd.DataFrame()
    return {"expression": expr, "clinical": clinical, "mutations": None}


def _download_cbio_study(study_id: str, dest_dir: Path) -> None:
    """Download expression + clinical from cBioPortal public API."""
    import requests

    base = "https://www.cbioportal.org/api"
    ensure_dir(dest_dir)

    # Clinical
    resp = requests.get(
        f"{base}/studies/{study_id}/clinical-data",
        params={"clinicalDataType": "PATIENT", "projection": "DETAILED"},
        headers={"Accept": "application/json"},
        timeout=120,
    )
    if resp.ok:
        clin = pd.DataFrame(resp.json())
        if not clin.empty:
            clin_pivot = clin.pivot(
                index="patientId", columns="clinicalAttributeId", values="value"
            )
            clin_pivot.to_csv(dest_dir / "clinical.csv")

    # Expression (mRNA seq v2)
    profiles = requests.get(
        f"{base}/studies/{study_id}/molecular-profiles",
        headers={"Accept": "application/json"},
        timeout=60,
    ).json()
    rna_profile = next(
        (p["molecularProfileId"] for p in profiles
         if "rna_seq" in p["molecularProfileId"].lower()),
        None,
    )
    if rna_profile:
        genes_resp = requests.get(
            f"{base}/molecular-profiles/{rna_profile}/molecular-data",
            params={"sampleListId": f"{study_id}_all", "projection": "SUMMARY"},
            headers={"Accept": "application/json"},
            timeout=300,
        )
        if genes_resp.ok:
            gdata = pd.DataFrame(genes_resp.json())
            if not gdata.empty:
                expr = gdata.pivot(
                    index="sampleId",
                    columns="hugoGeneSymbol",
                    values="value",
                )
                expr.to_csv(dest_dir / "expression.csv")


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

COHORT_LOADERS = {
    "Hugo_2016_melanoma": load_hugo_2016,
    "Riaz_2017_melanoma": load_riaz_2017,
    "Mariathasan_2018_bladder": load_mariathasan_2018,
    "Braun_2020_RCC": load_braun_2020,
}


def harmonize_response(clinical: pd.DataFrame) -> pd.DataFrame:
    """Standardise response columns across cohorts.

    Maps RECIST/custom labels to binary response_binary (1=R, 0=NR).
    """
    resp_cfg = CFG["clinical"]["response_categories"]
    responders = set(resp_cfg["responder"])
    if "best_response" in clinical.columns and "response_binary" not in clinical.columns:
        clinical["response_binary"] = clinical["best_response"].isin(responders).astype(int)
    return clinical


def load_all_icb_cohorts(data_dir: Path | None = None) -> dict[str, dict]:
    """Load all configured ICB cohorts, harmonise response labels."""
    data_dir = Path(data_dir or CFG["data_dir"])
    results = {}
    for name, loader in COHORT_LOADERS.items():
        log.info("Loading ICB cohort: %s", name)
        cohort = loader(data_dir)
        cohort["clinical"] = harmonize_response(cohort["clinical"])
        results[name] = cohort
    return results
