"""Pointwise confidence bands for mean persistence summaries.

The band is non-parametric: at each time point t, the empirical ``alpha/2`` and
``1 - alpha/2`` quantiles of the bootstrap curves. Bands are display-only — they
shade plots but never enter scores or p-values.
"""

from __future__ import annotations

import numpy as np


def _bootstrap_matrix(
    bootstrapped: list[np.ndarray], t_common: np.ndarray
) -> np.ndarray:
    """Interpolate each bootstrap curve onto the mean curve's t-grid."""
    matrix = np.zeros((len(bootstrapped), len(t_common)))
    for i, curve in enumerate(bootstrapped):
        matrix[i, :] = np.interp(t_common, curve[:, 0], curve[:, 1], left=0.0, right=0.0)
    return matrix


def pointwise_band(
    bootstrapped: list[np.ndarray], mean_curve: np.ndarray, alpha: float = 0.05
) -> dict[str, np.ndarray]:
    """Empirical ``alpha/2`` and ``1 - alpha/2`` quantiles at each time point."""
    if len(bootstrapped) == 0:
        raise ValueError("Cannot compute a band from zero bootstrap curves.")
    t_common = mean_curve[:, 0]
    matrix = _bootstrap_matrix(bootstrapped, t_common)
    lower = np.quantile(matrix, alpha / 2, axis=0)
    upper = np.quantile(matrix, 1 - alpha / 2, axis=0)
    return {"lower_bound": lower, "upper_bound": upper, "half_bound": (upper - lower) / 2}
