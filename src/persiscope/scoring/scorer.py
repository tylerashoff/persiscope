"""The scoring entry point: compare two representations, or a whole list.

``Scorer`` unifies two families under one interface:

- the **distribution score** (``energy``) compares the two bootstrap *sets*
  of summary curves directly;
- **curve metrics** (``euclidean``, ``js``, ``wasserstein``, ...) compare the
  two *mean* curves.

In both cases an optional permutation p-value is available.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np

from .distance_metrics import get_curve_metric
from .landscape_scores import energy_statistic
from .permutation import permutation_pvalue
from .results import ScoreMatrix, ScoreResult

RandomState = int | np.random.Generator | None

_DISTRIBUTION_SCORES = {"energy": energy_statistic}


class Scorer:
    """Score pairs of representations under a chosen method and summary.

    Parameters
    ----------
    method:
        ``"energy"`` (the distribution score), or any curve metric
        name (``"euclidean"``, ``"cosine"``, ``"dtw"``, ``"frechet"``,
        ``"spectral"``, ``"chi_squared"``, ``"kl"``, ``"js"``, ``"wasserstein"``).
    summary:
        ``"landscape"`` or ``"silhouette"``.
    run_pvalue, n_permutations:
        Whether to run the permutation test and how many permutations to use.
    random_state:
        Seed or generator for the permutation draws.
    """

    def __init__(
        self,
        method: str = "energy",
        summary: str = "landscape",
        run_pvalue: bool = False,
        n_permutations: int = 100,
        random_state: RandomState = None,
    ):
        if summary not in ("landscape", "silhouette"):
            raise ValueError(f"summary must be 'landscape' or 'silhouette', got {summary!r}.")
        self.method = method
        self.summary = summary
        self.run_pvalue = run_pvalue
        self.n_permutations = n_permutations
        self.random_state = random_state
        self._is_distribution = method in _DISTRIBUTION_SCORES

    def score(self, rep_a, rep_b) -> ScoreResult:
        """Score the pair ``(rep_a, rep_b)``."""
        rng = np.random.default_rng(self.random_state)

        if self._is_distribution:
            score_func = _DISTRIBUTION_SCORES[self.method]
            set1 = rep_a.bootstrapped(self.summary)
            set2 = rep_b.bootstrapped(self.summary)
            observed = score_func(set1, set2, None)
            perm_scores, pvalue = (None, None)
            if self.run_pvalue:
                perm_scores, pvalue = permutation_pvalue(
                    observed, score_func, set1, set2,
                    n_permutations=self.n_permutations, average_groups=False, rng=rng,
                )
        else:
            metric = get_curve_metric(self.method)
            mean1 = rep_a.summary(self.summary)
            mean2 = rep_b.summary(self.summary)
            observed = metric(mean1, mean2, None)
            perm_scores, pvalue = (None, None)
            if self.run_pvalue:
                set1 = rep_a.bootstrapped(self.summary)
                set2 = rep_b.bootstrapped(self.summary)
                perm_scores, pvalue = permutation_pvalue(
                    observed, metric, set1, set2,
                    n_permutations=self.n_permutations, average_groups=True, rng=rng,
                )

        return ScoreResult(
            score=float(observed),
            method=self.method,
            summary=self.summary,
            pvalue=pvalue,
            perm_scores=perm_scores,
        )

    def score_all(
        self, representations: Sequence, labels: list[Any] | None = None
    ) -> ScoreMatrix:
        """Score every ``C(k, 2)`` pair into a symmetric matrix."""
        reps = list(representations)
        k = len(reps)
        if k < 2:
            raise ValueError("Need at least 2 representations to compare.")
        if labels is None:
            labels = [getattr(r, "label", None) or i for i, r in enumerate(reps)]

        matrix = np.zeros((k, k))
        pvalues = np.full((k, k), np.nan) if self.run_pvalue else None

        for i, j in combinations(range(k), 2):
            result = self.score(reps[i], reps[j])
            matrix[i, j] = matrix[j, i] = result.score
            if pvalues is not None:
                pvalues[i, j] = pvalues[j, i] = result.pvalue

        return ScoreMatrix(
            matrix=matrix,
            method=self.method,
            summary=self.summary,
            pvalues=pvalues,
            labels=labels,
        )


def score(rep_a, rep_b, method: str = "energy", summary: str = "landscape", **kwargs) -> ScoreResult:
    """Functional wrapper: score a single pair."""
    return Scorer(method=method, summary=summary, **kwargs).score(rep_a, rep_b)


def score_matrix(
    representations: Sequence,
    method: str = "energy",
    summary: str = "landscape",
    labels: list[Any] | None = None,
    **kwargs,
) -> ScoreMatrix:
    """Functional wrapper: all-pairs score matrix over a list of representations."""
    return Scorer(method=method, summary=summary, **kwargs).score_all(representations, labels=labels)
