"""Pairwise score-matrix heatmap."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._backends import require_plotly


def _matrix_and_labels(result):
    """Extract ``(matrix, labels)`` from a ComparisonResult, ScoreMatrix, or array."""
    if hasattr(result, "matrix"):
        labels = getattr(result, "labels", None)
        return np.asarray(result.matrix, dtype=float), labels
    return np.asarray(result, dtype=float), None


def plot_score_heatmap(
    result: Any | np.ndarray,
    *,
    labels: list[Any] | None = None,
    colorscale: str = "Viridis",
    title: str | None = None,
):
    """Heatmap of a pairwise score matrix.

    Accepts a :class:`~persiscope.scoring.results.ComparisonResult`, a
    :class:`~persiscope.scoring.results.ScoreMatrix`, or a raw ``(k, k)`` array.
    Returns a plotly ``Figure``.
    """
    go = require_plotly()
    matrix, found_labels = _matrix_and_labels(result)
    if labels is None:
        labels = found_labels
    if labels is None:
        labels = [str(i) for i in range(matrix.shape[0])]
    else:
        labels = [str(v) for v in labels]

    method = getattr(result, "method", None)
    summary = getattr(result, "summary", None)
    subtitle = f" — {method}/{summary}" if method else ""

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=colorscale,
            colorbar={"title": "score"},
        )
    )
    fig.update_layout(
        title=title or f"Pairwise dissimilarity{subtitle}",
        xaxis={"side": "bottom"},
        yaxis={"autorange": "reversed"},
        width=520,
        height=480,
    )
    return fig
