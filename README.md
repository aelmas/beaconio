# BEACON-IO: Expression-Driven Dependencies at the Tumor-Immune Interface for Precision Immuno-Oncology

[![CI](https://github.com/aelmas/beacon-io/actions/workflows/ci.yml/badge.svg)](https://github.com/aelmas/beacon-io/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![BEACON paper](https://img.shields.io/badge/BEACON-GigaScience%202026-green.svg)](https://doi.org/10.1093/gigascience/giag011)

## The Problem

Immune checkpoint blockade (ICB) has transformed oncology, yet **60-80% of patients fail to respond**. Current biomarkers — PD-L1 expression, tumor mutational burden (TMB), and microsatellite instability — are insufficient predictors of response and provide no mechanistic insight into *why* tumors resist immunotherapy or *what to combine with ICB* to overcome resistance.

The field urgently needs:
1. **Mechanistic biomarkers** that explain *why* a tumor is immune-resistant (not just whether it correlates with response)
2. **Druggable targets** for rational combination strategies (ICB + targeted therapy)
3. **Multi-omic evidence** integrating functional dependencies, immune microenvironment, and clinical outcomes

## The Insight

[BEACON](https://doi.org/10.1093/gigascience/giag011) (Elmas et al., *GigaScience* 2026) demonstrated that cancer cells are preferentially dependent on genes they highly express — "expression-driven dependencies" (EDD). These represent precision vulnerabilities: tumors that overexpress a gene *need* that gene to survive.

**BEACON-IO extends this principle to the tumor-immune interface.** We hypothesize that:

> Tumors that evade immune attack are specifically dependent on genes driving immune evasion programmes. These immune-context-specific EDDs represent rational targets for combination with ICB — targeting the very dependencies that enable immune escape.

## Scope of the submitted manuscript vs. planned roadmap

> **What the manuscript actually delivers** (and is reproduced by the pipeline):
> BEACON MCMC EDD across **24 DepMap 24Q4 lineages** (CRISPR + RNA-seq); immune
> stratification by a **mean ESTIMATE-immune-gene score** (not full deconvolution);
> a **lineage-corrected solid-tumor differential EDD** with a posterior-difference
> test; **PRISM 24Q2** drug-sensitivity correlation; **TCGA** survival (10 cancer
> types); a **single** ICB cohort (Hugo 2016, *n*=28) used only to show the
> signature is *not* a response predictor; **MSigDB Hallmark** enrichment; and a
> **five-tier** evidence integration (E1–E4, E6).
>
> **Planned / not in this release** (scaffolding present in `config/`): MCPcounter/
> CIBERSORTx deconvolution, proteomics, a 10-cohort ICB meta-analysis (tier E5),
> single-cell compartment resolution (tier E7), and DGIdb druggability (tier E8).
> The Aims below describe the full roadmap; items beyond the delivered scope above
> are future work.

## Three Specific Aims (full roadmap)

### Aim 1: Identify immune-context-specific expression-driven dependencies
- Run BEACON across 17 cancer lineages (DepMap: ~1000 cell lines, CRISPR + RNA-seq + proteomics)
- Stratify by immune microenvironment (ESTIMATE, MCPcounter deconvolution)
- Identify **differential EDDs**: genes with stronger expression-dependency coupling in immune-hot vs immune-cold contexts
- Correlate EDD strength with immune evasion programmes (antigen presentation loss, TGF-β exclusion, Wnt/β-catenin, myeloid suppression)

### Aim 2: Discover rational ICB combination targets via drug-sensitivity integration
- Map immune-specific EDDs to PRISM drug-sensitivity profiles (~1500 compounds x ~500 cell lines)
- Identify drugs whose sensitivity scales with EDD target expression → direct pharmacological handle
- Cross-reference with DGIdb/DrugBank for approved/clinical-stage drugs
- Rank combination candidates by composite score (EDD strength + drug sensitivity + druggability + immune specificity)

### Aim 3: Validate BEACON-IO targets against clinical ICB response and survival
- **TCGA survival**: test EDD targets against overall survival across 10 ICB-relevant cancer types (n>10,000)
- **ICB response prediction**: build a BEACON-IO gene signature and benchmark against TMB, PD-L1, GEP (Ayers 2017), IMPRES, and TIDE across 10 public ICB cohorts (~1,500 patients total)
- **Meta-analysis**: fixed-effects inverse-variance meta-analysis of BEACON-IO target associations across cohorts
- **Single-cell resolution**: use public scRNA-seq from ICB-treated tumors (Jerby-Arnon 2018, Sade-Feldman 2018, Zhang 2021 pan-cancer T cell atlas) to resolve whether targets are tumour-intrinsic vs immune vs stromal

## Public Datasets

| Dataset | Source | Samples | Data Types |
|---------|--------|---------|------------|
| DepMap 24Q4 | depmap.org | ~1,000 cell lines | CRISPR (Chronos), RNA-seq, proteomics (MS), CN, mutations |
| PRISM Repurposing | depmap.org | ~500 cell lines | Drug sensitivity (~1,500 compounds) |
| TCGA | Xena/GDC | ~11,000 tumors (10 types) | RNA-seq, WES, clinical, survival |
| Hugo 2016 | GSE78220 | 28 melanoma | RNA-seq, anti-PD1 response |
| Riaz 2017 | GSE91061 | 51 melanoma | RNA-seq, nivolumab response |
| Liu 2019 | cBioPortal | 144 melanoma | RNA-seq, WES, anti-PD1 response |
| Van Allen 2015 | dbGaP | 110 melanoma | WES, anti-CTLA4 response |
| Braun 2020 | cBioPortal | 592 RCC | RNA-seq, WES, anti-PD1 response |
| Mariathasan 2018 | IMvigor210 | 348 bladder | RNA-seq, atezolizumab response |
| Kim 2018 | GSE135222 | 45 gastric | RNA-seq, anti-PD1 response |
| Cho 2020 | GSE126044 | 42 NSCLC | RNA-seq, anti-PD1 response |
| Jerby-Arnon 2018 | GSE115978 | 7,000 cells (31 tumors) | scRNA-seq, melanoma ICB |
| Sade-Feldman 2018 | GSE120575 | 16,000 TILs (48 samples) | scRNA-seq, melanoma ICB |
| Zhang 2021 | Zenodo | 400K cells (316 patients) | scRNA-seq, pan-cancer T cell atlas |

## Analysis Pipeline

```
01_download ─────────► 02_beacon_edd (BEACON MCMC + fast screen)
                             │
              ┌──────────────┼─────────────────┐
              ▼              ▼                  ▼
        03_immune       04_combination     06_singlecell
        (TME deconv,    (PRISM drugs,      (cell-type
         evasion,        DGIdb,             resolution,
         diff EDD)       combo scoring)     compartment)
              │              │                  │
              └──────────────┼─────────────────┘
                             ▼
                    05_clinical_validation
                    (TCGA survival, ICB response,
                     biomarker benchmark, meta-analysis)
                             │
                             ▼
                      07_integration
                      (8-tier evidence scoring,
                       ranked target catalogue)
```

### Evidence Integration Framework

Each gene is scored across 8 evidence tiers (inspired by Open Targets):

| Tier | Evidence | Weight | Source |
|------|----------|--------|--------|
| E1 | Expression-driven dependency (BEACON) | 0.15 | DepMap CRISPR + expression |
| E2 | Immune-context specificity | 0.15 | Differential EDD (hot vs cold) |
| E3 | Immune evasion correlation | 0.10 | EDD ~ evasion programme scores |
| E4 | Drug sensitivity | 0.15 | PRISM correlation |
| E5 | Clinical ICB response | 0.20 | Meta-analysis across ICB cohorts |
| E6 | TCGA survival association | 0.10 | Cox PH across cancer types |
| E7 | Tumour-intrinsic (sc-confirmed) | 0.05 | scRNA-seq compartment |
| E8 | Druggability | 0.10 | DGIdb / DrugBank |

## Installation

```bash
# Clone
git clone https://github.com/aelmas/beacon-io.git
cd beacon-io

# Install (Python 3.10.13 is the pinned/tested version; see requirements-lock.txt)
pip install -e ".[all]"
# For a bit-reproducible environment matching the manuscript:
#   pip install -r requirements-lock.txt

# Or minimal (no single-cell / GPU)
pip install -e .
```

## Usage

```bash
# Full pipeline via Snakemake
snakemake --cores 8

# Fast screen only (Spearman, no MCMC)
snakemake --cores 4 --config beacon_mode=fast

# Skip single-cell (if data not yet downloaded)
snakemake --cores 4 --config skip_singlecell=True

# Individual steps
python scripts/01_download_data.py
python scripts/02_beacon_edd.py --mode fast
python scripts/03_immune_context.py
python scripts/04_combination_targets.py
python scripts/05_clinical_validation.py
python scripts/06_singlecell.py
python scripts/07_integration.py

# Dry run
snakemake -n
```

## Expected Outputs

```
analysis/out/
├── beacon_edd/
│   ├── beacon_edd_all_lineages.csv       # Full BEACON results (gene x lineage)
│   ├── beacon_edd_significant.csv        # FDR < 0.05, rho < -0.25
│   ├── fast_screen_mrna.csv              # Spearman pre-filter
│   └── fast_screen_protein.csv
├── immune_context/
│   ├── differential_edd_hot_vs_cold.csv  # Immune-specific EDDs
│   ├── evasion_correlated_edd.csv        # EDD ~ evasion programmes
│   ├── tcga_estimate_scores.csv
│   └── tcga_evasion_scores.csv
├── combination/
│   ├── prism_edd_drug_hits.csv           # Gene-drug sensitivity pairs
│   ├── dgidb_annotations.csv             # Druggability
│   └── icb_combination_candidates.csv    # Ranked combinations
├── clinical/
│   ├── tcga_survival_results.csv         # Per-cancer Cox PH
│   ├── icb_response_aucs.csv             # BEACON-IO prediction AUCs
│   ├── icb_biomarker_benchmark.csv       # vs TMB, PD-L1, GEP, IMPRES
│   └── icb_meta_analysis.csv             # Cross-cohort meta
├── singlecell/
│   ├── compartment_consensus.csv         # Tumour vs immune vs stromal
│   └── *_diff_expr_R_vs_NR.csv           # Per-dataset DE
└── integration/
    ├── beacon_io_evidence_table.csv      # Full evidence matrix
    ├── beacon_io_top30_targets.csv       # Top ranked targets
    └── summary_statistics.csv
```

## Why This Matters

This project addresses three critical gaps in immuno-oncology:

1. **From correlation to mechanism**: Unlike TMB or PD-L1, BEACON-IO identifies targets that tumors *functionally depend on* for immune evasion — not just markers that correlate with response.

2. **Actionable combination targets**: Each BEACON-IO target comes with drug-sensitivity evidence and druggability annotation, providing a direct path to combination clinical trials.

3. **Multi-scale validation**: Cell-line functional dependencies → bulk tumour immune context → patient ICB response → single-cell resolution. This is the first framework to systematically bridge expression-driven dependencies with the tumor immune microenvironment across all these scales.

## Key Methodological Innovations Over Original BEACON

| Feature | BEACON (Elmas 2026) | BEACON-IO (this work) |
|---------|---------------------|----------------------|
| Scope | Pan-cancer target discovery | Immuno-oncology focused |
| Immune context | None | TME deconvolution + evasion programmes |
| Stratification | By lineage only | By lineage × immune status |
| Drug integration | DGIdb enrichment only | PRISM sensitivity + DGIdb + combination scoring |
| Clinical validation | None (cell-line only) | 10 ICB cohorts (~1,500 patients) + TCGA survival |
| Single-cell | None | scRNA-seq compartment resolution |
| Evidence framework | Per-gene rho + FDR | 8-tier weighted composite score |

## Citation

If you use BEACON-IO, please cite:

```
Elmas A, et al. Expression-driven genetic dependency reveals targets for
precision oncology. GigaScience. 2026;15:giag011.
```

## License

MIT
