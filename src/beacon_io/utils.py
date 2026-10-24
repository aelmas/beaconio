"""Shared utilities: logging, I/O helpers, gene-ID mapping."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def align_matrices(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """Align DataFrames to shared rows (samples) and columns (genes)."""
    shared_rows = dfs[0].index
    shared_cols = dfs[0].columns
    for df in dfs[1:]:
        shared_rows = shared_rows.intersection(df.index)
        shared_cols = shared_cols.intersection(df.columns)
    return [df.loc[shared_rows, shared_cols] for df in dfs]


def filter_low_variance(df: pd.DataFrame, quantile: float = 0.1) -> pd.DataFrame:
    """Drop columns (genes) with variance below the given quantile."""
    variances = df.var(axis=0)
    threshold = variances.quantile(quantile)
    return df.loc[:, variances > threshold]


def fdr_correction(pvalues: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """Benjamini-Hochberg FDR correction (handles NaN p-values)."""
    from statsmodels.stats.multitest import multipletests

    pvals = np.asarray(pvalues, dtype=float)
    result = np.full_like(pvals, np.nan)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return result
    _, corrected, _, _ = multipletests(pvals[valid], method=method)
    result[valid] = corrected
    return result
