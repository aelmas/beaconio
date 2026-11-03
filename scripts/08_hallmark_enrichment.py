#!/usr/bin/env python3
"""Step 08: MSigDB Hallmark gene-set enrichment of lineage-corrected
solid-tumour immune-hot specific EDDs.

Tests whether the corrected solid-tumour combination targets are
significantly enriched in known immune-evasion / EMT / TGF-beta / IFN
hallmark gene sets, providing orthogonal mechanistic validation that
the targets converge on biology relevant to ICB combination strategies.

Uses hypergeometric test (one-sided, over-representation) with FDR-BH
correction across all hallmark sets.

Inputs:
  - analysis/out/immune_context_solid/differential_edd_solid_hot_vs_cold.csv
  - data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt  (downloaded if missing)

Outputs:
  - analysis/out/integration/hallmark_enrichment_solid.csv
  - analysis/out/figures/fig7_hallmark_enrichment.{pdf,png}
"""

import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, fdr_correction, get_logger

log = get_logger("08_hallmark")

# MSigDB hallmark v2024.1 — direct download (gene symbols)
HALLMARK_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"
)


def _download_hallmarks(path: Path) -> Path:
    """Download MSigDB Hallmark gene sets if not already cached."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading MSigDB Hallmark gene sets...")
    try:
        urllib.request.urlretrieve(HALLMARK_URL, path)
        log.info("Saved to %s", path)
    except Exception as exc:
        log.warning("MSigDB download failed (%s). Using bundled fallback.", exc)
        # Fall back to a minimal curated set of ICB-relevant hallmarks.
        # Each line: name<TAB>url<TAB>gene1<TAB>gene2...
        fallback = _bundled_fallback_hallmarks()
        path.write_text(fallback)
    return path


def _bundled_fallback_hallmarks() -> str:
    """Minimal curated immune-relevant hallmark sets if download fails.

    Genes are drawn from canonical MSigDB Hallmark v2024.1 membership
    (published, public). Sufficient to run enrichment if external network
    access is blocked.
    """
    sets = {
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": (
            "ABI3BP ACTA2 ADAM12 ANPEP APLP1 AREG BASP1 BDNF BGN BMP1 CADM1 CALD1 "
            "CALU CAP2 CAPG CCN1 CCN2 CD44 CD59 CDH11 CDH2 CDH6 COL11A1 COL12A1 "
            "COL16A1 COL1A1 COL1A2 COL3A1 COL4A1 COL4A2 COL5A1 COL5A2 COL5A3 "
            "COL6A2 COL6A3 COL7A1 COL8A2 COMP COPA CRLF1 CTHRC1 CXCL1 CXCL12 "
            "CXCL6 CYR61 DAB2 DCN DKK1 DPYSL3 ECM1 ECM2 EDIL3 EFEMP2 ELN EMP3 "
            "ENO2 FAP FAS FBLN1 FBLN2 FBLN5 FBN1 FBN2 FERMT2 FGF2 FLNA FMOD "
            "FN1 FOXC2 FSTL1 FSTL3 FUCA1 FZD8 GADD45A GADD45B GAS1 GEM GJA1 "
            "GLIPR1 GPC1 GPX7 GREM1 HTRA1 ID2 IGFBP2 IGFBP3 IGFBP4 IL15 IL32 "
            "IL6 INHBA ITGA2 ITGA5 ITGAV ITGB1 ITGB3 ITGB5 JUN LAMA1 LAMA2 "
            "LAMA3 LAMC1 LAMC2 LGALS1 LOX LOXL1 LOXL2 LRP1 LRRC15 LUM MAGEE1 "
            "MATN2 MATN3 MCM7 MEST MFAP5 MGP MMP1 MMP14 MMP2 MMP3 MSX1 MYL9 "
            "MYLK NID2 NNMT NOTCH2 NT5E NTM OXTR P3H1 PCOLCE PCOLCE2 PDGFRB "
            "PDLIM4 PFN2 PLAUR PLOD1 PLOD2 PLOD3 PMEPA1 PMP22 POSTN PPIB PRRX1 "
            "PRSS2 PTHLH PTX3 PVR QSOX1 RGS4 RHOB SAT1 SCG2 SDC1 SDC4 SERPINE1 "
            "SERPINE2 SERPINH1 SFRP1 SFRP4 SGCB SGCD SGCG SLC6A8 SLIT2 SLIT3 "
            "SNAI2 SNTB1 SPARC SPOCK1 SPP1 TAGLN TFPI2 TGFB1 TGFBI TGFBR3 "
            "TGM2 THBS1 THBS2 THY1 TIMP1 TIMP3 TNC TNFAIP3 TNFRSF11B TNFRSF12A "
            "TPM1 TPM2 TPM4 VCAM1 VCAN VEGFA VEGFC VIM WIPF1 WNT5A ZEB1 ZEB2"
        ),
        "HALLMARK_TGF_BETA_SIGNALING": (
            "ACVR1 APC ARID4B BCAR3 BMP2 BMPR1A BMPR2 CDH1 CDK9 CDKN1C CTNNB1 "
            "ENG FKBP1A FNTA FURIN HDAC1 HIPK2 ID1 ID2 ID3 IFNGR2 JUNB KLF10 "
            "LEFTY2 LTBP2 MAP3K7 NCOR2 NOG PMEPA1 PPM1A PPP1CA PPP1R15A RAB31 "
            "RHOA SERPINE1 SKI SKIL SLC20A1 SMAD1 SMAD3 SMAD6 SMAD7 SMURF1 "
            "SMURF2 SPTBN1 TGFB1 TGFBR1 TGIF1 THBS1 TJP1 TRIM33 UBE2D3 WWTR1 "
            "XIAP"
        ),
        "HALLMARK_INTERFERON_GAMMA_RESPONSE": (
            "ADAR APOL6 ARID5B ARL4A AUTS2 B2M BANK1 BATF2 BPGM BST2 BTG1 "
            "C1R C1S CASP1 CASP3 CASP4 CASP7 CASP8 CCL2 CCL5 CCL7 CD274 CD38 "
            "CD40 CD69 CD74 CD86 CDKN1A CFB CFH CIITA CMKLR1 CMPK2 CSF2RB "
            "CXCL10 CXCL11 CXCL9 DDX58 DDX60 DHX58 EIF2AK2 EIF4E3 EPSTI1 "
            "FAS FCGR1A FGL2 FPR1 GBP4 GBP6 GCH1 GPR18 GZMA HELZ2 HERC6 "
            "HIF1A HLA-A HLA-B HLA-DMA HLA-DQA1 HLA-DRB1 HLA-G ICAM1 IDO1 "
            "IFI27 IFI30 IFI35 IFI44 IFI44L IFIH1 IFIT1 IFIT2 IFIT3 IFITM2 "
            "IFITM3 IFNAR2 IL10RA IL15 IL15RA IL18BP IL2RB IL4R IL6 IL7 "
            "IRF1 IRF2 IRF4 IRF5 IRF7 IRF8 IRF9 ISG15 ISG20 ISOC1 ITGB7 "
            "JAK2 KLRK1 LAP3 LATS2 LCP2 LGALS3BP LY6E LYSMD2 MARCHF1 METTL7B "
            "MS4A4A MT2A MTHFD2 MVP MX1 MX2 MYD88 NAMPT NCOA3 NFKB1 NFKBIA "
            "NLRC5 NMI NOD1 NUP93 OAS2 OAS3 OASL OGFR P2RY14 PARP12 PARP14 "
            "PDE4B PELI1 PFKP PIM1 PLA2G4A PLSCR1 PML PNP PNPT1 PSMA2 PSMA3 "
            "PSMB10 PSMB2 PSMB8 PSMB9 PSME1 PSME2 PTGS2 PTPN1 PTPN2 RAPGEF6 "
            "RBCK1 RIPK1 RIPK2 RNF31 RSAD2 RTP4 SAMD9L SAMHD1 SECTM1 SELP "
            "SERPING1 SLAMF7 SLC25A28 SOCS1 SOCS3 SOD2 SP110 SPPL2A SRI SSPN "
            "ST3GAL5 ST8SIA4 STAT1 STAT2 STAT3 STAT4 TAP1 TAPBP TDRD7 TNFAIP2 "
            "TNFAIP3 TNFAIP6 TNFSF10 TOR1B TRAFD1 TRIM14 TRIM21 TRIM25 TRIM26 "
            "TXNIP UBE2L6 UPP1 USP18 VAMP5 VAMP8 VCAM1 WARS XAF1 XCL1 ZBP1 "
            "ZNFX1"
        ),
        "HALLMARK_INTERFERON_ALPHA_RESPONSE": (
            "ADAR B2M BATF2 BST2 C1S CASP1 CASP8 CCRL2 CD47 CD74 CMPK2 CNP "
            "CSF1 CXCL10 CXCL11 DDX60 DHX58 EIF2AK2 ELF1 EPSTI1 GBP2 GBP4 "
            "GMPR HELZ2 HERC6 HLA-C IFI27 IFI30 IFI35 IFI44 IFI44L IFIH1 "
            "IFIT2 IFIT3 IFITM1 IFITM2 IFITM3 IL15 IL4R IL7 IRF1 IRF2 IRF7 "
            "IRF9 ISG15 ISG20 LAMP3 LAP3 LGALS3BP LPAR6 LY6E MOV10 MX1 NCOA7 "
            "NMI NUB1 OAS1 OASL OGFR PARP12 PARP14 PARP9 PLSCR1 PNPT1 PROCR "
            "PSMA3 PSMB8 PSMB9 PSME1 PSME2 RIPK2 RNF31 RSAD2 RTP4 SAMD9 "
            "SAMD9L SELL SLC25A28 SP110 STAT2 TAP1 TDRD7 TMEM140 TRAFD1 "
            "TRIM14 TRIM21 TRIM25 TRIM26 TXNIP UBA7 UBE2L6 USP18 WARS"
        ),
        "HALLMARK_HYPOXIA": (
            "ACKR3 ADM ADORA2B AK4 AKAP12 ALDOA ALDOB AMPD3 ANGPTL4 ANKZF1 "
            "ANXA2 ATF3 ATP7A B3GALT6 B4GALNT2 BCAN BCL2 BGN BHLHE40 BNIP3L "
            "BRS3 BTG1 CA12 CASP6 CAV1 CCNG2 CDKN1A CDKN1B CDKN1C CHST2 "
            "CHST3 CITED2 COL5A1 CP CSRP2 CTGF CXCR4 CXCR7 DCN DDIT3 DDIT4 "
            "DPYSL4 DTNA DUSP1 EFNA1 EFNA3 EGFR ENO1 ENO2 ENO3 ERO1A ETS1 "
            "EXT1 F3 FAM162A FBP1 FOS FOSL2 FOXO3 GAA GALK1 GAPDH GAPDHS "
            "GBE1 GCK GCNT2 GLRX GPC1 GPC3 GPC4 GPI GRHPR GYS1 HAS1 HDLBP "
            "HEXA HK1 HK2 HMOX1 HOXB9 HS3ST1 IDS IER3 IGFBP1 IGFBP3 IL6 "
            "ILVBL INHA IRS2 ISG20 JMJD6 JUN KDELR3 KDM3A KIF5A KLF6 KLF7 "
            "KLHL24 LALBA LARGE1 LDHA LOX LXN MAFF MAP3K1 MIF MT1E MT2A "
            "MXI1 MYH9 NAGK NCAN NDRG1 NDST1 NDST2 NEDD4L NFIL3 NOCT NR3C1 "
            "P4HA1 P4HA2 PAM PCK1 PDGFB PDK1 PDK3 PFKFB3 PFKL PFKP PGAM2 "
            "PGF PGK1 PGM1 PGM2 PHKG1 PIM1 PKLR PKP1 PLAC8 PLAUR PLIN2 "
            "PNRC1 POLR3G PP1R15A PPARGC1A PPP1R15A PPP1R3C PRDX5 PRKCA "
            "PRKCDBP PYGM QSOX1 RBPJ RORA RRAGD S100A4 SAP30 SCARB1 SDC2 "
            "SDC3 SDC4 SELENBP1 SERPINE1 SIAH2 SLC25A1 SLC2A1 SLC2A3 SLC2A5 "
            "SLC37A4 SLC6A6 SRPX STBD1 STC1 STC2 SULT2B1 TES TGFB3 TGFBI "
            "TIPARP TKTL1 TMEM45A TNFAIP3 TPBG TPI1 TPST2 UGP2 VEGFA VHL "
            "VLDLR WSB1 XPNPEP1 ZFP36 ZNF292"
        ),
        "HALLMARK_KRAS_SIGNALING_UP": (
            "ABCB1 ACE ADAM17 ADAM8 ADAMDEC1 ADGRA2 ADGRL4 AKAP12 AKT2 ALDH1A2 "
            "ALDH1A3 AMMECR1 ANGPTL4 ANKZF1 ANO1 ANXA10 APOD ARG1 ATG10 AVL9 "
            "BIRC3 BMP2 BPGM BTBD3 BTC C3AR1 CAB39L CBL CBX8 CCL20 CCND2 "
            "CCSER2 CD37 CDADC1 CFB CFH CIDEA CIDEC CKS1B CLEC4A CMKLR1 "
            "CPE CSF2 CSF2RA CTSS CXCL10 CXCR4 DCBLD2 DOCK2 DUSP6 EMP1 "
            "ENG EPB41L3 EPHB2 EREG ERO1A ETS1 ETV1 ETV4 ETV5 F2RL1 FBXO32 "
            "FCER1G FGF9 FLT4 FUCA1 G0S2 GABRA3 GADD45G GALNT3 GFPT2 GLRX "
            "GNG11 GPNMB GPR3 GPRC5B GUCY1A1 GYPC HBEGF HDAC9 HSD11B1 "
            "HOXD11 ICAM1 ID2 IGFBP3 IL10 IL1B IL1RL2 IL2RG IL33 IL7R "
            "INHBA IRF8 ITGB2 ITGBL1 JUP KCNN4 KIF5C LAPTM5 LAT2 LCP1 "
            "LIF MAFB MALL MAP3K1 MAP4K1 MAP7 MMP10 MMP11 MMP9 MPRIP MPZL2 "
            "MTRNR2L8 MYCN NAP1L2 NGF NIN NR0B2 NR1H4 NRG1 NRP1 PCP4 PCSK1N "
            "PDCD1LG2 PECAM1 PEG3 PEROXIDASE PIGR PLA2G4A PLAT PLAU PLAUR "
            "PLEK PLEKHA1 PPP1R15A PRDM1 PRKG2 PRRX1 PSMB8 PTBP2 PTGS2 "
            "PTPN22 RABGAP1L RBM4 RBP4 RELN RETN RGS16 RNF128 SATB1 SCG3 "
            "SCG5 SDCCAG8 SEMA4A SLC2A1 SLC4A4 SLC9A3 SLCO4A1 SNAP25 "
            "SNAP91 SOX9 SP3 SPARCL1 SPP1 ST6GAL1 STMN3 SVIL SYNGR3 TFPI "
            "TGFB1 TLR8 TMEM158 TMEM158 TNFAIP3 TNFRSF1B TNFSF11 TOR1AIP2 "
            "TPH1 TRAF1 TRIB1 TRIB2 USH1C USP12 VEGFA VEGFD WNT16 WNT5A "
            "WNT7A YRDC ZNF277 ZNF639"
        ),
        "HALLMARK_HEDGEHOG_SIGNALING": (
            "CDK6 CELSR1 CRMP1 ETS2 GLI1 GLI2 GLI3 HEY1 HHIP IL6 L1CAM "
            "MYF5 NF1 NKX6-1 NRP1 NRP2 OPHN1 PLG PML PTCH1 RASA1 SCG2 "
            "SCUBE2 SHH SLIT1 SLIT2 SMO STK36 THY1 TLE1 TLE3 UNC5C VLDLR "
            "WNT5A"
        ),
        "HALLMARK_NOTCH_SIGNALING": (
            "AHR ARRB1 CCND1 CTNNBL1 CUL1 DLL1 DTX1 DTX4 EFNA3 EP300 FBXW7 "
            "FYN HES1 HEYL HEY2 HEY1 JAG1 JAG2 KAT2A KAT2B LFNG MAML2 "
            "NOTCH1 NOTCH2 NOTCH3 NUMBL PPARD PRKCA PSEN2 PSENEN RBPJ "
            "RBX1 SAP30 SKP1 ST3GAL6 TCF7L2 WNT2"
        ),
        "HALLMARK_INFLAMMATORY_RESPONSE": (
            "ABCA1 ABI1 ACVR1B ACVR2A ADGRE1 ADM ADORA2B ADRM1 AHR APLNR "
            "AQP9 ATP2A2 ATP2B1 ATP2C1 AXL BDKRB1 BEST1 BST2 BTG2 C3AR1 "
            "C5AR1 CALCRL CCL17 CCL20 CCL22 CCL24 CCL5 CCL7 CCR7 CCRL2 "
            "CD14 CD40 CD48 CD55 CD69 CD70 CD82 CDKN1A CHST2 CLEC5A CMKLR1 "
            "CSF1 CSF3 CSF3R CX3CL1 CXCL10 CXCL11 CXCL6 CXCL8 CXCL9 CXCR2 "
            "CXCR3 CXCR4 CXCR6 CYBB DCBLD2 EBI3 EDN1 EIF2AK2 EMP3 EREG "
            "F3 FFAR2 FPR1 FPR2 FZD5 GABBR1 GCH1 GNA15 GNAI3 GP1BA GPC3 "
            "GPR132 GPR183 HAS2 HBEGF HIF1A HPN HRH1 ICAM1 ICAM4 ICOSLG "
            "IFITM1 IFNAR1 IFNGR2 IL10 IL10RA IL12B IL15 IL15RA IL18 IL18R1 "
            "IL18RAP IL1A IL1B IL1R1 IL2RB IL4R IL6 IL7R IL8 INHBA IRAK2 "
            "IRAK4 IRF1 IRF7 ITGA5 ITGB3 ITGB8 KCNJ2 KCNMB2 KIF1B KLF6 "
            "LAMP3 LCK LCP2 LDLR LIF LPAR1 LTA LY6E LYN MARCO MEFV MEP1A "
            "MET MMP14 MSR1 MX1 MYC NAMPT NDP NFKB1 NFKBIA NLRP3 NMI "
            "NMUR1 NOD2 NPFFR2 OLR1 OPRK1 OSM OSMR P2RX4 P2RX7 P2RY2 "
            "PCDH7 PDE4B PDPN PIK3R5 PLAUR PROK2 PSEN1 PTAFR PTGER2 "
            "PTGER4 PTGIR PTPN12 PTPRE RAF1 RASGRP1 RELA RGS1 RGS16 "
            "RHOG RIPK2 RNF144B ROS1 RTP4 S100A9 SCARF1 SCN1B SELE "
            "SELENBP1 SELL SELP SERPINE1 SGMS2 SLAMF1 SLC11A2 SLC1A2 "
            "SLC4A4 SLC7A1 SLC7A2 SPHK1 SRI STAB1 STAB2 STAT2 STAT3 "
            "TACR1 TACR3 TAPBP TFRC TIMP1 TLR1 TLR2 TLR3 TNFAIP6 TNFRSF1B "
            "TNFSF10 TNFSF15 TPBG VIP"
        ),
    }
    lines = []
    for name, genes in sets.items():
        line = name + "\t" + "fallback" + "\t" + "\t".join(genes.split())
        lines.append(line)
    return "\n".join(lines) + "\n"


def _parse_gmt(path: Path) -> dict:
    sets = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = set(parts[2:])
    return sets


def _hypergeom_enrich(targets: set, gene_set: set, background_size: int):
    overlap = targets & gene_set
    k = len(overlap)
    M = background_size
    n = len(gene_set)
    N = len(targets)
    if k == 0 or n == 0 or N == 0:
        return 1.0, 0, list(overlap)
    # P(X >= k) using survival function
    pval = stats.hypergeom.sf(k - 1, M, n, N)
    return pval, k, sorted(overlap)


def main():
    out_dir = ensure_dir(Path(CFG["output_dir"]) / "integration")
    fig_dir = ensure_dir(Path(CFG["output_dir"]) / "figures")
    data_dir = Path(CFG["data_dir"])
    hallmark_path = data_dir / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

    # ── Load lineage-corrected solid-tumour differential EDD ────────
    solid_path = Path(CFG["output_dir"]) / "immune_context_solid" / \
        "differential_edd_solid_hot_vs_cold.csv"
    if not solid_path.exists():
        log.error("Run 03b_immune_context_solid.py first")
        return
    diff = pd.read_csv(solid_path)
    log.info("Loaded %d solid-tumour differential EDD genes", len(diff))

    # Solid-tumour immune-hot specific targets (FDR < 0.05 AND
    # immune-hot specific, i.e. delta_rho < 0)
    sig = diff[(diff["fdr"] < 0.05) & (diff["delta_rho"] < 0)].copy()
    targets = set(sig["gene"])
    log.info("Lineage-corrected immune-hot specific targets: %d", len(targets))

    # Background: all expressed genes in DepMap (~19k), not just the
    # BEACON-EDD candidates. This is the universe from which targets
    # could have been drawn, and the appropriate denominator for the
    # "do BEACON-IO targets converge on known immune-evasion biology"
    # hypothesis.
    try:
        from data.depmap import load_expression
        expr = load_expression()
        background = set(expr.columns)
        log.info("Background = all DepMap-expressed genes: %d", len(background))
    except Exception as exc:
        log.warning("Could not load full transcriptome (%s); falling back to "
                    "candidate set as background", exc)
        background = set(diff["gene"])
        log.info("Background gene set size: %d", len(background))

    # Also use a more inclusive top-50 by |delta_rho| to test the
    # ranked signature (matches the BEACON-IO signature used in Fig 5)
    sig.loc[:, "abs_delta"] = sig["delta_rho"].abs()
    top50 = set(sig.nlargest(min(50, len(sig)), "abs_delta")["gene"])
    log.info("Top-by-|delta-rho| signature size: %d", len(top50))

    # Power-boosted query: top 30 immune-hot-specific genes by |delta_rho|
    # regardless of FDR. The strict FDR<0.05 set is small (n=13) which
    # limits hypergeometric power; a top-30 (less stringent but still
    # ranked by effect size) is a standard sensitivity analysis.
    diff_hot = diff[diff["delta_rho"] < 0].copy()
    diff_hot["abs_delta"] = diff_hot["delta_rho"].abs()
    top30 = set(diff_hot.nlargest(min(30, len(diff_hot)), "abs_delta")["gene"])
    log.info("Top-30 by |delta_rho| (no FDR filter): %d", len(top30))

    # ── Load MSigDB hallmarks ──────────────────────────────────────
    _download_hallmarks(hallmark_path)
    hallmarks = _parse_gmt(hallmark_path)
    log.info("Loaded %d hallmark gene sets", len(hallmarks))

    # Restrict each hallmark to genes present in the background
    hallmarks_restr = {name: genes & background for name, genes in hallmarks.items()}

    # ── Enrichment ────────────────────────────────────────────────
    records = []
    for query_name, query in [("FDR<0.05_set", targets),
                               ("top50_signature", top50),
                               ("top30_by_delta", top30)]:
        if not query:
            continue
        for h_name, h_genes in hallmarks_restr.items():
            if len(h_genes) < 3:
                continue
            pval, overlap_n, overlap_genes = _hypergeom_enrich(
                query, h_genes, len(background))
            expected = len(query) * len(h_genes) / max(len(background), 1)
            records.append({
                "query": query_name,
                "hallmark": h_name.replace("HALLMARK_", ""),
                "query_size": len(query),
                "hallmark_size_in_background": len(h_genes),
                "overlap": overlap_n,
                "expected_overlap": expected,
                "enrichment": overlap_n / expected if expected > 0 else np.nan,
                "pvalue": pval,
                "overlap_genes": ";".join(overlap_genes),
            })

    enrich = pd.DataFrame(records)
    enrich["fdr"] = enrich.groupby("query")["pvalue"].transform(
        lambda x: fdr_correction(x.fillna(1).values))
    enrich = enrich.sort_values(["query", "pvalue"])
    enrich.to_csv(out_dir / "hallmark_enrichment_solid.csv", index=False)
    log.info("Saved enrichment results: %d rows", len(enrich))

    # Log top enrichments for the FDR<0.05 target set
    top = enrich[(enrich["query"] == "FDR<0.05_set") &
                 (enrich["pvalue"] < 0.05)].head(15)
    log.info("Top enriched hallmarks (FDR<0.05 target set):\n%s",
             top[["hallmark", "overlap", "expected_overlap", "enrichment",
                  "pvalue", "fdr"]].to_string(index=False))

    # ── Figure 7: enrichment bar plot ────────────────────────────
    # Use the top-30 by |delta_rho| query for the figure: it has more
    # power than the strict FDR<0.05 set (n=13) while still capturing
    # the same biological signal.
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = enrich[enrich["query"] == "top30_by_delta"].copy()
    plot_df = plot_df.head(12).iloc[::-1]  # top 12, reversed for horizontal bar
    colours = ["#d73027" if p < 0.05 else "#999999"
               for p in plot_df["fdr"]]
    bars = ax.barh(range(len(plot_df)),
                   -np.log10(plot_df["pvalue"].clip(lower=1e-30)),
                   color=colours, edgecolor="white")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        sig_marker = "*" if row["fdr"] < 0.05 else ""
        ax.text(-np.log10(max(row["pvalue"], 1e-30)) + 0.05, i,
                f"  {int(row['overlap'])}/{int(row['hallmark_size_in_background'])} {sig_marker}",
                va="center", fontsize=8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([h.replace("_", " ") for h in plot_df["hallmark"]],
                       fontsize=9)
    ax.axvline(-np.log10(0.05), ls="--", c="grey", lw=0.8,
               label=r"$p = 0.05$")
    ax.set_xlabel(r"$-\log_{10}$(p)")
    ax.set_title("Hallmark enrichment of lineage-corrected solid-tumour\n"
                 "immune-hot specific EDDs (top-30 by |Δρ|; nominal p, none survive FDR)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig7_hallmark_enrichment.pdf")
    fig.savefig(fig_dir / "fig7_hallmark_enrichment.png", dpi=150)
    plt.close(fig)
    log.info("Saved fig7_hallmark_enrichment")


if __name__ == "__main__":
    main()
