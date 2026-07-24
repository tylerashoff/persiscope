"""Result containers for scoring and comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ScoreResult:
    """A single pairwise score, optionally with a permutation p-value."""

    score: float
    method: str
    summary: str
    pvalue: float | None = None
    perm_scores: np.ndarray | None = None

    def __repr__(self) -> str:
        p = "" if self.pvalue is None else f", p={self.pvalue:.4g}"
        return f"ScoreResult({self.method}/{self.summary}: {self.score:.4g}{p})"


def _to_frame(matrix: np.ndarray, labels: list[Any] | None):
    """Labeled ``pandas.DataFrame`` view of a square matrix (lazy pandas import)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "to_frame() requires pandas. Install it with `pip install persiscope[frame]`."
        ) from exc
    idx = labels if labels is not None else list(range(matrix.shape[0]))
    return pd.DataFrame(matrix, index=idx, columns=idx)


@dataclass
class ScoreMatrix:
    """All-pairs score matrix over a list of representations."""

    matrix: np.ndarray  # (k, k) symmetric, zero diagonal
    method: str
    summary: str
    pvalues: np.ndarray | None = None  # (k, k) or None
    labels: list[Any] | None = None

    def to_frame(self):
        """Score matrix as a labeled ``pandas.DataFrame``."""
        return _to_frame(self.matrix, self.labels)

    def pvalue_frame(self):
        """P-value matrix as a labeled ``pandas.DataFrame`` (or ``None``)."""
        return None if self.pvalues is None else _to_frame(self.pvalues, self.labels)


@dataclass
class ComparisonResult:
    """Output of :func:`persiscope.compare`.

    Carries the all-pairs score matrix plus the fitted representations, so the
    (potentially expensive) topological transforms can be reused.
    """

    matrix: np.ndarray  # (k, k) symmetric, zero diagonal
    method: str
    summary: str
    pvalues: np.ndarray | None = None
    labels: list[Any] | None = None
    representations: list[Any] = field(default_factory=list, repr=False)

    def to_frame(self):
        """Score matrix as a labeled ``pandas.DataFrame``."""
        return _to_frame(self.matrix, self.labels)

    def pvalue_frame(self):
        """P-value matrix as a labeled ``pandas.DataFrame`` (or ``None``)."""
        return None if self.pvalues is None else _to_frame(self.pvalues, self.labels)

    def as_score_matrix(self) -> ScoreMatrix:
        """Drop the representations, keeping just the matrices and labels."""
        return ScoreMatrix(
            matrix=self.matrix,
            method=self.method,
            summary=self.summary,
            pvalues=self.pvalues,
            labels=self.labels,
        )

    def __repr__(self) -> str:
        return (
            f"ComparisonResult({self.method}/{self.summary}, "
            f"k={self.matrix.shape[0]}, pvalues={'yes' if self.pvalues is not None else 'no'})"
        )
