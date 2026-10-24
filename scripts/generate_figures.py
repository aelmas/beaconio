#!/usr/bin/env python3
"""Generate all publication figures for BEACON-IO.

Produces Figs 1-6 as PDF and PNG in analysis/out/figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from beacon_io.utils import get_logger

log = get_logger("figures")

OUT = Path("analysis/out")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = sns.color_palette("Set2", 10)


# =========================================================================
# Fig 1: Pan-cancer EDD landscape heatmap
# =========================================================================
def fig1_edd_heatmap():
    log.info("Fig 1: Pan-cancer EDD heatmap")
    all_edd = pd.read_csv(OUT / "beacon_edd/beacon_edd_all_lineages.csv")
    sig = all_edd[all_edd["significant"]].copy()

    # Top 30 genes by number of lineages
    top_genes = (
        sig.groupby("gene")["lineage"]
        .nunique()
        .sort_values(ascending=False)
        .head(30)
        .index.tolist()
    )

    # Build heatmap matrix: gene x lineage -> rho
    lineages = sig["lineage"].unique()
    matrix = pd.DataFrame(index=top_genes, columns=sorted(lineages), dtype=float)
    for _, row in sig[sig["gene"].isin(top_genes)].iterrows():
        matrix.loc[row["gene"], row["lineage"]] = row["rho_posterior_median"]

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = LinearSegmentedColormap.from_list("edd", ["#2166ac", "#f7f7f7", "#b2182b"])
    sns.heatmap(
        matrix.astype(float),
        cmap=cmap,
        center=0,
        vmin=-0.8,
        vmax=0.2,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "BEACON posterior rho (expression ~ dependency)", "shrink": 0.6},
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title("Expression-Driven Dependencies Across Cancer Lineages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cancer Lineage")
    ax.set_ylabel("Gene")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)

    # Add significance markers
    for i, gene in enumerate(matrix.index):
        for j, lin in enumerate(matrix.columns):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center", fontsize=6, color="white" if val < -0.4 else "black")

    fig.savefig(FIG / "fig1_edd_heatmap.pdf")
    fig.savefig(FIG / "fig1_edd_heatmap.png")
    plt.close(fig)
    log.info("  Saved fig1_edd_heatmap")

    # Fig 1B: Forest plot of top EDD genes with 95% HDI credible intervals
    if "rho_hdi_low" in all_edd.columns:
        top_sig = sig.sort_values("rho_posterior_median").drop_duplicates("gene").head(25)
        fig_b, ax_b = plt.subplots(figsize=(8, 10))
        y_pos = range(len(top_sig))
        ax_b.errorbar(
            top_sig["rho_posterior_median"], y_pos,
            xerr=[top_sig["rho_posterior_median"] - top_sig["rho_hdi_low"],
                  top_sig["rho_hdi_high"] - top_sig["rho_posterior_median"]],
            fmt="o", color="#2166ac", ecolor="#92c5de", elinewidth=2, capsize=3, markersize=5,
        )
        ax_b.axvline(0, ls="--", c="grey", lw=0.8)
        ax_b.axvline(-0.25, ls=":", c="red", lw=0.8, alpha=0.5, label="rho threshold")
        ax_b.set_yticks(list(y_pos))
        ax_b.set_yticklabels(top_sig["gene"].values, fontsize=8)
        ax_b.invert_yaxis()
        ax_b.set_xlabel("BEACON posterior rho (95% HDI)")
        ax_b.set_title("Top EDD Genes: Bayesian Posterior with 95% HDI", fontsize=13, fontweight="bold")
        # Add lineage labels
        for i, (_, row) in enumerate(top_sig.iterrows()):
            ax_b.text(row["rho_hdi_high"] + 0.02, i, row["lineage"],
                      fontsize=6, va="center", color="grey")
        ax_b.legend(fontsize=8)
        plt.tight_layout()
        fig_b.savefig(FIG / "fig1b_edd_forest_plot.pdf")
        fig_b.savefig(FIG / "fig1b_edd_forest_plot.png")
        plt.close(fig_b)
        log.info("  Saved fig1b_edd_forest_plot")


# =========================================================================
# Fig 2: Differential EDD — immune hot vs cold
# =========================================================================
def fig2_differential_edd():
    log.info("Fig 2: Differential EDD")
    diff = pd.read_csv(OUT / "immune_context/differential_edd_hot_vs_cold.csv")
    diff["neg_log10_fdr"] = -np.log10(diff["fdr"].clip(lower=1e-50))
    sig = diff["fdr"] < 0.05

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2A: Volcano plot
    ax = axes[0]
    ax.scatter(
        diff.loc[~sig, "delta_rho"],
        diff.loc[~sig, "neg_log10_fdr"],
        s=4, alpha=0.3, c="grey", label="NS",
    )
    hot_specific = sig & (diff["delta_rho"] < 0)
    cold_specific = sig & (diff["delta_rho"] > 0)
    ax.scatter(
        diff.loc[hot_specific, "delta_rho"],
        diff.loc[hot_specific, "neg_log10_fdr"],
        s=12, alpha=0.7, c="#d73027", label=f"Immune-hot specific ({hot_specific.sum()})",
    )
    ax.scatter(
        diff.loc[cold_specific, "delta_rho"],
        diff.loc[cold_specific, "neg_log10_fdr"],
        s=12, alpha=0.7, c="#4575b4", label=f"Immune-cold specific ({cold_specific.sum()})",
    )
    # Label top genes
    top = diff[sig].sort_values("neg_log10_fdr", ascending=False).head(10)
    for _, r in top.iterrows():
        ax.annotate(r["gene"], (r["delta_rho"], r["neg_log10_fdr"]),
                     fontsize=7, ha="center", va="bottom")
    ax.axhline(-np.log10(0.05), ls="--", c="grey", lw=0.8)
    ax.set_xlabel("Delta rho (hot - cold)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("A. Differential EDD: Immune-Hot vs Cold", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # 2B: Top 20 immune-specific genes barplot
    ax = axes[1]
    top20 = diff[sig].reindex(diff[sig]["delta_rho"].abs().sort_values(ascending=False).index).head(20)
    colors = ["#d73027" if d < 0 else "#4575b4" for d in top20["delta_rho"]]
    bars = ax.barh(range(len(top20)), top20["delta_rho"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["gene"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Delta rho (hot - cold)")
    ax.set_title("B. Top 20 Immune-Specific EDD Genes", fontweight="bold")
    ax.axvline(0, c="black", lw=0.8)

    plt.tight_layout()
    fig.savefig(FIG / "fig2_differential_edd.pdf")
    fig.savefig(FIG / "fig2_differential_edd.png")
    plt.close(fig)
    log.info("  Saved fig2_differential_edd")


# =========================================================================
# Fig 3: PRISM drug-target bubble plot
# =========================================================================
def fig3_prism_drugs():
    log.info("Fig 3: PRISM drug-target")
    prism = pd.read_csv(OUT / "combination/prism_edd_drug_hits.csv")
    sig_prism = prism[prism["fdr"] < 0.05].copy()

    # Top 15 EDD genes by strongest drug correlation
    top_genes = sig_prism.groupby("gene")["rho"].min().sort_values().head(15).index
    sub = sig_prism[sig_prism["gene"].isin(top_genes)]

    # For each gene, top 3 drugs
    rows = []
    for gene in top_genes:
        gene_hits = sub[sub["gene"] == gene].sort_values("rho").head(3)
        for _, r in gene_hits.iterrows():
            name = r.get("name", r["drug_id"])
            if pd.isna(name):
                name = r["drug_id"][:15]
            rows.append({"gene": gene, "drug": str(name)[:20], "rho": r["rho"], "n_lines": r["n_lines"]})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        plot_df["rho"],
        range(len(plot_df)),
        s=plot_df["n_lines"] * 0.3,
        c=plot_df["rho"],
        cmap="RdBu",
        vmin=-0.5,
        vmax=0.1,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_yticks(range(len(plot_df)))
    labels = [f"{r['gene']} - {r['drug']}" for _, r in plot_df.iterrows()]
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho (gene expression ~ drug sensitivity, PRISM)")
    ax.set_title("PRISM Drug Sensitivity Correlates of EDD Targets", fontsize=13, fontweight="bold")
    ax.axvline(0, c="grey", ls="--", lw=0.8)
    plt.colorbar(scatter, ax=ax, label="Rho", shrink=0.5)

    plt.tight_layout()
    fig.savefig(FIG / "fig3_prism_drugs.pdf")
    fig.savefig(FIG / "fig3_prism_drugs.png")
    plt.close(fig)
    log.info("  Saved fig3_prism_drugs")


# =========================================================================
# Fig 4: TCGA survival Kaplan-Meier curves (top 6 genes)
# =========================================================================
def fig4_survival():
    log.info("Fig 4: TCGA survival KM curves")
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    surv_results = pd.read_csv(OUT / "clinical/tcga_survival_results.csv")

    # Load expression + survival (reuse ensembl mapping)
    sys.path.insert(0, "src")
    from data.tcga import load_tcga_survival, load_tcga_clinical
    survival = load_tcga_survival()
    clinical = load_tcga_clinical()
    clinical["sample_15"] = clinical.index.str[:15]
    cmap_clin = clinical.set_index("sample_15")

    # Load mapped expression
    mapping = pd.read_csv(OUT / "clinical/ensembl_to_symbol.csv", index_col=0, header=None)
    mapping.columns = ["symbol"]
    ens2sym = dict(zip(mapping.index, mapping["symbol"]))

    expr_raw = pd.read_csv("data/raw/tcga/tcga_RSEM_gene_tpm.gz", sep="\t", index_col=0)
    new_idx = [ens2sym.get(g.split(".")[0], "") for g in expr_raw.index]
    expr_raw.index = new_idx
    expr_raw = expr_raw[expr_raw.index != ""]
    expr_raw = expr_raw[~expr_raw.index.duplicated(keep="first")]
    expr_full = expr_raw.T
    expr_full.index = [idx[:15] for idx in expr_full.index]
    expr_full = expr_full[~expr_full.index.duplicated(keep="first")]

    # Pick top 6 gene-cancer pairs
    top6 = surv_results.sort_values("cox_pvalue").head(6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()

    for idx, (_, row) in enumerate(top6.iterrows()):
        ax = axes[idx // 3, idx % 3]
        gene, cancer = row["gene"], row["cancer_type"]

        cancer_ids = cmap_clin[cmap_clin["cancer_type"] == cancer].index
        shared = cancer_ids.intersection(survival.index).intersection(expr_full.index)
        if len(shared) < 30 or gene not in expr_full.columns:
            ax.set_visible(False)
            continue

        surv = survival.loc[shared]
        expr = expr_full.loc[shared, gene].astype(float)
        median_val = expr.median()
        high = shared[expr >= median_val]
        low = shared[expr < median_val]

        T_h, E_h = surv.loc[high, "OS.time"].astype(float), surv.loc[high, "OS"].astype(float)
        T_l, E_l = surv.loc[low, "OS.time"].astype(float), surv.loc[low, "OS"].astype(float)

        # Drop NaN
        valid_h = T_h.notna() & E_h.notna()
        valid_l = T_l.notna() & E_l.notna()

        kmf_h.fit(T_h[valid_h], E_h[valid_h], label=f"{gene} high")
        kmf_l.fit(T_l[valid_l], E_l[valid_l], label=f"{gene} low")

        kmf_h.plot_survival_function(ax=ax, ci_show=False, color="#d73027", lw=2)
        kmf_l.plot_survival_function(ax=ax, ci_show=False, color="#4575b4", lw=2)

        try:
            lr = logrank_test(T_h[valid_h], T_l[valid_l], E_h[valid_h], E_l[valid_l])
            pval = lr.p_value
        except:
            pval = np.nan

        ax.set_title(f"{gene} - {cancer}\nHR={row['cox_hr']:.2f}, p={pval:.1e}", fontsize=10)
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival probability")
        ax.legend(fontsize=8)

    plt.suptitle("TCGA Overall Survival: Top BEACON-IO Targets", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG / "fig4_tcga_survival.pdf")
    fig.savefig(FIG / "fig4_tcga_survival.png")
    plt.close(fig)
    log.info("  Saved fig4_tcga_survival")


# =========================================================================
# Fig 5: ICB benchmark ROC / barplot
# =========================================================================
def fig5_icb_benchmark():
    """Fig 5: directional AUC with bootstrap 95% CIs.

    At n=28 / 4 responders no biomarker is statistically distinguishable; the
    figure shows DIRECTIONAL AUCs (no auto-flip) with wide, overlapping
    bootstrap CIs, making explicit that the cohort cannot adjudicate biomarker
    performance and that BEACON-IO is not positioned as a response predictor.
    """
    log.info("Fig 5: ICB benchmark (directional AUC + bootstrap CI)")
    ci_path = OUT / "clinical/icb_auc_bootstrap_ci.csv"
    if not ci_path.exists():
        log.warning("No bootstrap-CI data, skipping Fig 5")
        return
    ci = pd.read_csv(ci_path).sort_values("auc_directional")

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(ci))
    colours = ["#d73027" if "BEACON-IO" in b else "#4575b4" for b in ci["biomarker"]]
    err_low = (ci["auc_directional"] - ci["ci95_low"]).values
    err_high = (ci["ci95_high"] - ci["auc_directional"]).values
    ax.errorbar(ci["auc_directional"], y, xerr=[err_low, err_high], fmt="none",
                ecolor="#555555", elinewidth=1.2, capsize=4, zorder=1)
    ax.scatter(ci["auc_directional"], y, c=colours, s=90, zorder=2, edgecolors="white")
    for i, (_, r) in enumerate(ci.iterrows()):
        ax.text(r["ci95_high"] + 0.02, i,
                f"{r['auc_directional']:.2f} [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]  (n={int(r['n_genes'])})",
                va="center", fontsize=8)
    ax.axvline(0.5, ls="--", c="grey", lw=0.9, label="Chance (AUC = 0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels(ci["biomarker"], fontsize=9)
    ax.set_xlabel("Directional AUC (no auto-flip) with bootstrap 95% CI")
    ax.set_xlim(0, 1.15)
    ax.set_title(
        "ICB response prediction is uninformative at this sample size\n"
        "(Hugo 2016 melanoma, n=28; 4 R / 24 NR)",
        fontsize=11, fontweight="bold")
    ax.text(0.02, 0.02,
            "All 95% CIs overlap and span chance: no biomarker is\n"
            "statistically distinguishable. BEACON-IO (red) is a target-\n"
            "nomination framework, not a response predictor.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5e6",
                      edgecolor="#d73027", linewidth=0.8))
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIG / "fig5_icb_benchmark.pdf")
    fig.savefig(FIG / "fig5_icb_benchmark.png")
    plt.close(fig)
    log.info("  Saved fig5_icb_benchmark")


# =========================================================================
# Fig 6: Integrated evidence — multi-tier bubble/dot plot
# =========================================================================
def fig6_evidence_integration(evidence_path: Path | None = None,
                              out_suffix: str = "",
                              title_suffix: str = ""):
    log.info("Fig 6: Integrated evidence (suffix=%s)", out_suffix)
    if evidence_path is None:
        evidence_path = OUT / "integration/beacon_io_evidence_table.csv"
    evidence = pd.read_csv(evidence_path)
    top = evidence.head(30).copy()

    # Evidence presence matrix — only the five populated tiers (E5 ICB-meta,
    # E7 single-cell, E8 DGIdb were not available from public data).
    ev_tiers = {
        "E1: EDD": "E1_edd_rho",
        "E2: Immune-spec.": "E2_delta_rho",
        "E3: Evasion": "E3_evasion_rho",
        "E4: PRISM": "E4_prism_rho",
        "E6: Survival": "E6_min_cox_p",
    }

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5], wspace=0.05)

    # 6A: Composite score barplot
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.YlOrRd(top["composite_score"] / top["composite_score"].max())
    ax1.barh(range(len(top)), top["composite_score"], color=colors, edgecolor="white")
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top["gene"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Composite Score")
    ax1.set_title("A. Composite Score", fontweight="bold")

    # 6B: Evidence tier dot matrix
    ax2 = fig.add_subplot(gs[1])
    for j, (tier_name, col) in enumerate(ev_tiers.items()):
        if col not in top.columns:
            continue
        for i, (_, row) in enumerate(top.iterrows()):
            val = row.get(col, np.nan)
            if pd.notna(val) and val != 0:
                # Size proportional to effect
                if "rho" in col:
                    size = abs(val) * 200
                elif "cox" in col:
                    size = min(-np.log10(val + 1e-50) * 5, 200)
                elif "n_drugs" in col:
                    size = min(val * 3, 200)
                else:
                    size = 80
                color = "#d73027" if (("rho" in col and val < 0) or ("cox" in col)) else "#4575b4"
                ax2.scatter(j, i, s=size, c=color, alpha=0.7, edgecolors="black", linewidths=0.5)
            else:
                ax2.scatter(j, i, s=10, c="lightgrey", alpha=0.3)

    ax2.set_xticks(range(len(ev_tiers)))
    ax2.set_xticklabels(ev_tiers.keys(), rotation=45, ha="right", fontsize=9)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels([""] * len(top))
    ax2.invert_yaxis()
    ax2.set_title("B. Multi-Tier Evidence", fontweight="bold")

    # Grid lines
    for i in range(len(top)):
        ax2.axhline(i, c="lightgrey", lw=0.3)

    plt.suptitle(f"BEACON-IO: Integrated Target Ranking{title_suffix}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG / f"fig6_evidence_integration{out_suffix}.pdf")
    fig.savefig(FIG / f"fig6_evidence_integration{out_suffix}.png")
    plt.close(fig)
    log.info("  Saved fig6_evidence_integration%s", out_suffix)


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    log.info("=== Generating BEACON-IO publication figures ===")
    fig1_edd_heatmap()
    fig2_differential_edd()
    fig3_prism_drugs()
    fig4_survival()
    fig5_icb_benchmark()
    fig6_evidence_integration()
    log.info("All figures saved to %s", FIG)
