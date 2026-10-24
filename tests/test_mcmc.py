"""Slow MCMC recovery test (CI skips this via --ignore=tests/test_mcmc.py).

Validates that the BEACON bivariate-normal NUTS model recovers a known
correlation from synthetic data, within a tolerance appropriate to n and
the sampler settings.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_beacon_recovers_known_rho():
    from beacon_io.engine import _beacon_single_gene

    rng = np.random.default_rng(42)
    true_rho = -0.6
    n = 200
    cov = np.array([[1.0, true_rho], [true_rho, 1.0]])
    xy = rng.multivariate_normal([0, 0], cov, size=n)
    res = _beacon_single_gene(xy[:, 0], xy[:, 1], gene="SYN", lineage="test")
    # Posterior median should be within ~0.1 of the truth and HDI should cover it.
    assert abs(res.rho_posterior_median - true_rho) < 0.12
    assert res.rho_hdi_low <= true_rho <= res.rho_hdi_high
