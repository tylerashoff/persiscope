"""Curve distance metrics between two mean summary curves.

Each metric takes two ``(resolution, 2)`` arrays (``t, value``) and returns a
scalar. Metrics that need equal-length, common-support inputs interpolate both
curves onto a shared grid first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance

MetricFunc = Callable[[np.ndarray, np.ndarray, Any | None], float]

_EPS = 1e-10


def _extract_and_interpolate(array1: np.ndarray, array2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the two curves' y-values interpolated onto a shared x-grid."""
    from scipy.interpolate import interp1d

    def xy(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 2 and a.shape[1] == 2:
            return a[:, 0], a[:, 1]
        return np.arange(len(a)), a

    x1, y1 = xy(array1)
    x2, y2 = xy(array2)

    x_min = min(x1.min(), x2.min())
    x_max = max(x1.max(), x2.max())
    n_points = max(len(x1), len(x2))
    x_common = np.linspace(x_min, x_max, n_points)

    f1 = interp1d(x1, y1, kind="linear", fill_value=0.0, bounds_error=False)
    f2 = interp1d(x2, y2, kind="linear", fill_value=0.0, bounds_error=False)
    return f1(x_common), f2(x_common)


def _euclidean(a1, a2, config=None) -> float:
    y1, y2 = _extract_and_interpolate(a1, a2)
    return float(np.linalg.norm(y1 - y2))


def _cosine(a1, a2, config=None) -> float:
    y1, y2 = _extract_and_interpolate(a1, a2)
    n1, n2 = np.linalg.norm(y1), np.linalg.norm(y2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(y1, y2) / (n1 * n2))


def _dtw(a1, a2, config=None) -> float:
    def y(a):
        a = np.asarray(a, dtype=float)
        return a[:, 1] if a.ndim == 2 and a.shape[1] == 2 else a

    try:
        from dtaidistance import dtw
    except ImportError as exc:
        raise ImportError(
            "The 'dtw' metric needs dtaidistance. Install with `pip install persiscope[metrics]`."
        ) from exc
    return float(dtw.distance(y(a1), y(a2)))


def _frechet(a1, a2, config=None) -> float:
    def as2d(a):
        a = np.asarray(a, dtype=float)
        return a if a.ndim == 2 else np.column_stack([np.arange(len(a)), a])

    try:
        from similaritymeasures import frechet_dist
    except ImportError as exc:
        raise ImportError(
            "The 'frechet' metric needs similaritymeasures. "
            "Install with `pip install persiscope[metrics]`."
        ) from exc
    return float(frechet_dist(as2d(a1), as2d(a2)))


def _spectral(a1, a2, config=None) -> float:
    y1, y2 = _extract_and_interpolate(a1, a2)
    m1 = np.abs(np.fft.fft(y1))
    m2 = np.abs(np.fft.fft(y2))
    return float(np.linalg.norm(m1 - m2))


def _chi_squared(a1, a2, config=None) -> float:
    y1, y2 = _extract_and_interpolate(a1, a2)
    y1 = y1 - y1.min() + _EPS
    y2 = y2 - y2.min() + _EPS
    return float(np.sum(((y1 - y2) ** 2) / (y1 + y2)))


def _kl_divergence(a1, a2, config=None) -> float:
    y1, y2 = _extract_and_interpolate(a1, a2)
    y1 = y1 + _EPS
    y2 = y2 + _EPS
    p = y1 / np.sum(y1)
    q = y2 / np.sum(y2)
    return float(entropy(p, q))


def _js_distance(a1, a2, config=None) -> float:
    y1, y2 = _extract_and_interpolate(a1, a2)
    y1 = y1 + _EPS
    y2 = y2 + _EPS
    p = y1 / np.sum(y1)
    q = y2 / np.sum(y2)
    return float(jensenshannon(p, q))


def _wasserstein(a1, a2, config=None) -> float:
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    return float(wasserstein_distance(a1[:, 0], a2[:, 0], a1[:, 1], a2[:, 1]))


_METRICS = {
    "euclidean": _euclidean,
    "euclidean_distance": _euclidean,
    "cosine": _cosine,
    "cosine_distance": _cosine,
    "dtw": _dtw,
    "dtw_distance": _dtw,
    "frechet": _frechet,
    "frechet_distance": _frechet,
    "spectral": _spectral,
    "spectral_distance": _spectral,
    "chi_squared": _chi_squared,
    "chi_squared_distance": _chi_squared,
    "kl": _kl_divergence,
    "kl_divergence": _kl_divergence,
    "js": _js_distance,
    "js_distance": _js_distance,
    "wasserstein": _wasserstein,
    "wasserstein_distance": _wasserstein,
}

CURVE_METRICS = sorted(set(_METRICS))


def get_curve_metric(name: str) -> MetricFunc:
    """Return the curve-metric function registered under ``name``."""
    if name not in _METRICS:
        raise ValueError(f"Unknown curve metric {name!r}. Options: {CURVE_METRICS}.")
    return _METRICS[name]
