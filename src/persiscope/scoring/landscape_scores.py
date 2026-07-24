"""Distribution-level scores over sets of bootstrap summary curves.

The energy statistic takes two *sets* of curves (the bootstrap replicates of
two representations) and returns a scalar dissimilarity.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def energy_statistic(
    summaries1: list[np.ndarray],
    summaries2: list[np.ndarray],
    config: Any | None = None,
) -> float:
    """Energy distance ``2*between - within1 - within2`` between two curve sets.

    Zero when the two bootstrap distributions coincide, positive as they
    separate. Distances are Euclidean norms between whole ``(resolution, 2)``
    curves.
    """
    n, m = len(summaries1), len(summaries2)

    within1 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            within1 += np.linalg.norm(summaries1[i] - summaries1[j])
    within1 = within1 * 2 / (n**2)

    within2 = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            within2 += np.linalg.norm(summaries2[i] - summaries2[j])
    within2 = within2 * 2 / (m**2)

    between = 0.0
    for i in range(n):
        for j in range(m):
            between += np.linalg.norm(summaries1[i] - summaries2[j])
    between /= n * m

    return float(2 * between - within1 - within2)
