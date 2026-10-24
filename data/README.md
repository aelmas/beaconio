# Data Sources

All raw data used by BEACON-IO is publicly available and downloaded automatically
by the pipeline. Raw files are excluded from this repository (see `.gitignore`).

## Automatic download

Run step 01 to download everything:

```bash
python scripts/01_download_data.py
# or via Snakemake:
snakemake --cores 4 download_data
```

## Manual sources

| Dataset | Source | Version | Size |
|---------|--------|---------|------|
| DepMap CRISPR gene effect | [DepMap 24Q4](https://depmap.org/portal/download/) via Figshare | 24Q4 | ~200 MB |
| DepMap expression (TPM) | DepMap 24Q4 via Figshare | 24Q4 | ~300 MB |
| DepMap cell line info | DepMap 24Q4 via Figshare | 24Q4 | ~5 MB |
| PRISM repurposing (LFC) | [PRISM 24Q2](https://depmap.org/repurposing/) via Figshare | 24Q2 | ~450 MB |
| TCGA expression (RSEM TPM) | [UCSC Xena toil hub](https://xenabrowser.net/) | GDC/Toil | ~700 MB |
| TCGA clinical phenotype | [UCSC Xena GDC hub](https://xenabrowser.net/) | GDC | ~5 MB |
| TCGA survival | UCSC Xena toil hub | — | ~2 MB |
| Hugo 2016 melanoma ICB | [GEO: GSE78220](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220) | — | ~50 MB |
| Riaz 2017 melanoma ICB | [GEO: GSE91061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE91061) | — | ~40 MB |

## Directory layout after download

```
data/
├── raw/
│   ├── depmap/          # DepMap 24Q4 files
│   ├── prism/           # PRISM 24Q2 drug sensitivity
│   ├── tcga/            # TCGA expression + clinical + survival
│   └── icb/             # ICB cohort GEO downloads
└── processed/           # intermediate files (auto-generated)
```

## Licenses

- **DepMap**: CC BY 4.0
- **TCGA**: NCI GDC Data Access (open-access tier)
- **PRISM**: CC BY 4.0
- **GEO datasets**: individual study licenses; all open access
