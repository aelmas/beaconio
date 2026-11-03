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

import re
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
# GEO series-matrix metadata (authoritative per-sample response labels)
# ---------------------------------------------------------------------------

def _geo_series_metadata(geo_id: str, dest_dir: Path) -> pd.DataFrame:
    """Parse per-sample metadata from a GEO series matrix.

    This is the authoritative source of clinical annotation for a GEO
    series. Response labels must always be read from here -- never
    inferred from the column order of an expression matrix, which carries
    no clinical information.

    Returns a DataFrame indexed by GEO sample title, with one column per
    ``!Sample_characteristics_ch1`` field (named from its ``key: value``
    prefix), plus ``geo_accession``.
    """
    import gzip

    import requests

    ensure_dir(dest_dir)
    local = dest_dir / f"{geo_id}_series_matrix.txt.gz"
    if not local.exists():
        url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/"
               f"{geo_id}/matrix/{geo_id}_series_matrix.txt.gz")
        log.info("Downloading GEO series matrix for %s", geo_id)
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        local.write_bytes(resp.content)

    titles: list[str] = []
    accessions: list[str] = []
    char_rows: list[list[str]] = []
    with gzip.open(local, "rt", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            fields = [f.strip().strip('"') for f in line.rstrip("\n").split("\t")]
            tag, values = fields[0], fields[1:]
            if tag == "!Sample_title":
                titles = values
            elif tag == "!Sample_geo_accession":
                accessions = values
            elif tag == "!Sample_characteristics_ch1":
                char_rows.append(values)

    if not titles:
        raise ValueError(f"No !Sample_title in series matrix for {geo_id}")

    meta = pd.DataFrame(index=pd.Index(titles, name="sample_title"))
    if accessions:
        meta["geo_accession"] = accessions
    for values in char_rows:
        keys = {v.split(":", 1)[0].strip() for v in values if ":" in v}
        if len(keys) != 1:
            continue  # heterogeneous characteristics row -- cannot name a column
        key = keys.pop()
        col = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        meta[col] = [v.split(":", 1)[1].strip() if ":" in v else pd.NA
                     for v in values]
    return meta


def _response_column(meta: pd.DataFrame, geo_id: str) -> str:
    col = next((c for c in meta.columns if "response" in c), None)
    if col is None:
        raise ValueError(
            f"{geo_id} series matrix has no response characteristic; "
            f"available: {meta.columns.tolist()}"
        )
    return col


def _entrez_to_symbol(cache_dir: Path) -> dict:
    """Map NCBI Entrez GeneID -> HGNC symbol via NCBI ``gene_info``."""
    import requests

    ensure_dir(cache_dir)
    local = cache_dir / "Homo_sapiens.gene_info.gz"
    if not local.exists():
        url = ("https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
               "Homo_sapiens.gene_info.gz")
        log.info("Downloading NCBI gene_info for Entrez->symbol mapping")
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        local.write_bytes(resp.content)
    info = pd.read_csv(local, sep="\t", usecols=["GeneID", "Symbol"])
    return dict(zip(info["GeneID"].astype(str), info["Symbol"]))


# ---------------------------------------------------------------------------
# Cohort-specific loaders
# ---------------------------------------------------------------------------

# RECIST groupings. Responder = objective response (CR/PR); non-responder =
# progressive disease. Stable disease is ambiguous and excluded by default so
# that the two melanoma cohorts use the same contrast (Hugo has no SD arm).
_RESPONDER = {"Complete Response", "Partial Response", "PRCR", "CR", "PR"}
_NONRESPONDER = {"Progressive Disease", "PD"}
_STABLE = {"Stable Disease", "SD"}


def _binarise(detailed: pd.Series, include_sd: bool) -> pd.Series:
    def label(value):
        if value in _RESPONDER:
            return "R"
        if value in _NONRESPONDER:
            return "NR"
        if include_sd and value in _STABLE:
            return "NR"
        return None
    return detailed.map(label)


def load_hugo_2016(data_dir: Path, include_sd: bool = False) -> dict:
    """Hugo et al. 2016, melanoma anti-PD1, GSE78220.

    Response labels come from the ``anti-pd-1 response`` characteristic of
    the GEO series matrix: Complete/Partial Response are responders,
    Progressive Disease non-responders. The cohort is 15 R / 13 NR.

    Note Pt27A and Pt27B are two biopsies of the same patient; the
    ``patient`` column is provided so downstream analyses can account for
    that non-independence.
    """
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

    # Expression samples are "Pt1.baseline" / "Pt16.OnTx"; GEO titles are "Pt1".
    expr.index = expr.index.astype(str)
    sample_title = pd.Series(expr.index.str.split(".").str[0], index=expr.index)
    # Pt27A / Pt27B are two biopsies of patient Pt27 -- collapse the suffix.
    patient = sample_title.str.replace(r"^(Pt\d+)[A-Z]$", r"\1", regex=True)

    meta = _geo_series_metadata("GSE78220", geo_dir)
    resp_map = meta[_response_column(meta, "GSE78220")].to_dict()
    detailed = sample_title.map(resp_map)
    if detailed.isna().any():
        log.warning("Hugo 2016: no GEO response label for %s",
                    list(expr.index[detailed.isna()]))

    clinical = pd.DataFrame({
        "patient": patient,
        "response_detailed": detailed,
        "response": _binarise(detailed, include_sd),
    })
    clinical = clinical[clinical["response"].notna()].copy()
    clinical["response_binary"] = (clinical["response"] == "R").astype(int)
    expr = expr.loc[clinical.index]
    log.info("Hugo 2016: n=%d samples / %d patients (R=%d, NR=%d)",
             len(clinical), clinical["patient"].nunique(),
             int(clinical["response_binary"].sum()),
             int((1 - clinical["response_binary"]).sum()))
    return {"expression": expr, "clinical": clinical, "mutations": None}


def load_riaz_2017(data_dir: Path, visit: str = "Pre",
                   include_sd: bool = False) -> dict:
    """Riaz et al. 2017, melanoma nivolumab, GSE91061.

    Uses FPKM (not raw counts) so the expression scale matches Hugo 2016,
    maps the Entrez row index to HGNC symbols, and restricts to a single
    visit (``Pre`` by default) so that pre-treatment biopsies are compared
    like for like. Response labels come from the GEO ``response``
    characteristic: PRCR are responders, PD non-responders; SD and UNK are
    dropped unless ``include_sd``.
    """
    geo_dir = _download_geo_supp("GSE91061", data_dir / "icb" / "riaz_2017")
    fpkm_path = geo_dir / "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz"
    if not fpkm_path.exists():
        log.warning("Riaz 2017: FPKM file not found in %s", geo_dir)
        return {"expression": pd.DataFrame(), "clinical": pd.DataFrame(), "mutations": None}

    expr = pd.read_csv(fpkm_path, index_col=0)  # Entrez GeneID x samples
    symbols = _entrez_to_symbol(data_dir / "reference")
    expr.index = pd.Index([symbols.get(str(g)) for g in expr.index])
    expr = expr[expr.index.notna()]
    expr = expr[~expr.index.duplicated(keep="first")].T  # samples x genes

    meta = _geo_series_metadata("GSE91061", geo_dir)
    detailed = pd.Series(expr.index.map(meta[_response_column(meta, "GSE91061")]),
                         index=expr.index)
    visit_col = next((c for c in meta.columns if c.startswith("visit")), None)
    visits = (pd.Series(expr.index.map(meta[visit_col]), index=expr.index)
              if visit_col else pd.Series("Pre", index=expr.index))

    clinical = pd.DataFrame({
        "patient": pd.Series(expr.index.str.split("_").str[0], index=expr.index),
        "visit": visits,
        "response_detailed": detailed,
        "response": _binarise(detailed, include_sd),
    })
    keep = clinical["response"].notna() & (clinical["visit"] == visit)
    clinical = clinical[keep].copy()
    clinical["response_binary"] = (clinical["response"] == "R").astype(int)
    expr = expr.loc[clinical.index]
    log.info("Riaz 2017 (%s): n=%d samples / %d patients (R=%d, NR=%d)",
             visit, len(clinical), clinical["patient"].nunique(),
             int(clinical["response_binary"].sum()),
             int((1 - clinical["response_binary"]).sum()))
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
