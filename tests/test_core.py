"""Fast unit tests for BEACON-IO core utilities and evidence integration.

Addresses referee concern C19(d): the CI advertised a pytest suite but no
tests/ directory existed. These tests are deterministic and fast (no MCMC);
the slow MCMC recovery test lives in test_mcmc.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_fdr_correction_matches_statsmodels():
    """BEACON-IO fdr_correction must match statsmodels Benjamini-Hochberg."""
    from statsmodels.stats.multitest import multipletests

    from beacon_io.utils import fdr_correction

    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, 200)
    ours = fdr_correction(pvals)
    ref = multipletests(pvals, method="fdr_bh")[1]
    assert np.allclose(ours, ref, atol=1e-10)


def test_e2_is_hot_specific_directional():
    """E2 must credit immune-HOT-specific genes (negative delta_rho), not
    cold-specific ones (positive delta_rho) — the C2 fix."""
    from integration.evidence import compile_evidence

    edd = pd.DataFrame({
        "gene": ["HOT", "COLD"],
        "lineage": ["Lung", "Lung"],
        "rho_posterior_median": [-0.6, -0.6],
    })
    diff = pd.DataFrame({
        "gene": ["HOT", "COLD"],
        "delta_rho": [-0.4, +0.4],   # HOT is hot-specific, COLD is cold-specific
    })
    empty = pd.DataFrame()
    ev = compile_evidence(edd, diff, empty, empty, empty, empty, empty, empty)
    e2 = ev.set_index("gene")["E2_delta_rho"]
    # The directional tier stores the most-negative delta; HOT must rank better.
    assert e2["HOT"] < e2["COLD"]
    # And HOT must out-score COLD overall (everything else equal).
    comp = ev.set_index("gene")["composite_score"]
    assert comp["HOT"] > comp["COLD"]


def test_composite_in_unit_interval():
    """Composite score must lie in [0,1] after renormalization over populated
    tiers (the C18 fix), not be capped by unpopulated-tier weights."""
    from integration.evidence import compile_evidence

    edd = pd.DataFrame({
        "gene": [f"G{i}" for i in range(10)],
        "lineage": ["Lung"] * 10,
        "rho_posterior_median": np.linspace(-0.9, -0.3, 10),
    })
    diff = pd.DataFrame({"gene": [f"G{i}" for i in range(10)],
                         "delta_rho": np.linspace(-0.5, 0.1, 10)})
    empty = pd.DataFrame()
    ev = compile_evidence(edd, diff, empty, empty, empty, empty, empty, empty)
    assert ev["composite_score"].between(0, 1).all()


def test_solid_lineage_exclusion():
    """exclude_lineages must drop hematopoietic lineages from the E1 tier so a
    gene's E1 reflects its strongest SOLID dependency (the C5 fix)."""
    from integration.evidence import compile_evidence

    edd = pd.DataFrame({
        "gene": ["X", "X"],
        "lineage": ["Lymphoid", "Skin"],
        "rho_posterior_median": [-0.9, -0.4],  # strong in Lymphoid, weaker in Skin
    })
    empty = pd.DataFrame()
    ev = compile_evidence(edd, empty, empty, empty, empty, empty, empty, empty,
                          exclude_lineages=["Lymphoid", "Myeloid"])
    # Lymphoid excluded -> E1 must be the Skin value, not the Lymphoid value.
    assert np.isclose(ev.set_index("gene").loc["X", "E1_edd_rho"], -0.4)
