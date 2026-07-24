"""Permutation test for pairwise summary scores.

Pools the two groups' bootstrap replicates, repeatedly re-splits them at random,
recomputes the score, and reports the fraction of permutations at least as
extreme as the observed score. Larger scores mean *more different*, so the test
is one-sided.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm.auto import tqdm

ArrayLike = np.ndarray | list[np.ndarray]


def permutation_pvalue(
    observed_score: float,
    score_func: Callable[..., float],
    samples1: ArrayLike,
    samples2: ArrayLike,
    n_permutations: int = 100,
    config: Any | None = None,
    average_groups: bool = False,
    rng: np.random.Generator | None = None,
    progress: bool = False,
) -> tuple[np.ndarray, float]:
    """Return ``(perm_scores, p_value)``.

    Parameters
    ----------
    average_groups:
        If ``True`` each permuted group is averaged before scoring (used for
        curve metrics, which compare mean curves). If ``False`` the full
        pseudo-group set is passed through (the energy score).
    """
    if rng is None:
        rng = np.random.default_rng()

    s1 = np.asarray(samples1)
    s2 = np.asarray(samples2)
    if s1.ndim < 2 or s2.ndim < 2:
        raise ValueError(
            "Permutation inputs must be stackable arrays of shape (n_samples, ...) "
            "with ndim >= 2."
        )

    combined = np.concatenate([s1, s2], axis=0)
    n1 = len(s1)
    n_total = len(s1) + len(s2)

    perm_scores: list[float] = []
    count = 0
    iterator = range(n_permutations)
    if progress:
        iterator = tqdm(iterator, desc="permutation", leave=False)

    for _ in iterator:
        order = rng.permutation(n_total)
        g1 = combined[order[:n1]]
        g2 = combined[order[n1:]]
        if average_groups:
            g1 = np.mean(g1, axis=0)
            g2 = np.mean(g2, axis=0)
        perm_score = score_func(g1, g2, config)
        perm_scores.append(perm_score)
        if perm_score >= observed_score:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return np.array(perm_scores), p_value
