"""Persistence summary functions: tenting, landscapes, and silhouettes.

Given a (transformed) persistence diagram, each diagram point contributes a
triangular "tent". The landscape reads off the k-th largest tent at each point
in time; the silhouette is a persistence-weighted average of the tents. Both are
returned as ``(resolution, 2)`` arrays of ``(t, value)`` samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagram_transforms import DiagramTransform, RotateScale


def compute_tenting_values(
    transformed_diagram: np.ndarray, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tent function values for every diagram point over a shared time grid.

    The tent for a point ``(x, y)`` rises from 0 at ``t = x - y`` to a peak at
    ``t = x`` and back to 0 at ``t = x + y``.

    Returns ``(tenting_values, t_values)`` where ``tenting_values`` has shape
    ``(n_points, resolution)``.
    """
    transformed_diagram = np.asarray(transformed_diagram, dtype=float)
    if len(transformed_diagram) == 0:
        t_values = np.linspace(0.0, 0.0, resolution)
        return np.zeros((0, resolution)), t_values

    t_max = np.max(np.sum(transformed_diagram[:, [0, 1]], axis=1))
    t_values = np.linspace(0.0, t_max, resolution)

    x = transformed_diagram[:, 0][:, None]
    y = transformed_diagram[:, 1][:, None]
    t = t_values[None, :]

    rising = np.where((t >= x - y) & (t <= x), t - x + y, 0.0)
    falling = np.where((t > x) & (t <= x + y), x + y - t, 0.0)
    tenting_values = rising + falling
    return tenting_values, t_values


def compute_landscape(
    tenting_values: np.ndarray, t_values: np.ndarray, order: int = 0
) -> np.ndarray:
    """The ``order``-th persistence landscape: the ``order``-th largest tent per t."""
    if len(tenting_values) == 0:
        return np.column_stack([t_values, np.zeros_like(t_values)])
    # Descending sort along the point axis, then take row `order`.
    sorted_desc = np.sort(tenting_values, axis=0)[::-1]
    if order < sorted_desc.shape[0]:
        landscape_y = sorted_desc[order]
    else:
        landscape_y = np.zeros_like(t_values)
    return np.column_stack([t_values, landscape_y])


def compute_silhouette(
    tenting_values: np.ndarray,
    t_values: np.ndarray,
    diagram: np.ndarray,
    power: float = 3.0,
) -> np.ndarray:
    """Persistence-weighted silhouette (weights ``|death - birth|**power``)."""
    if len(tenting_values) == 0:
        return np.column_stack([t_values, np.zeros_like(t_values)])
    diagram = np.asarray(diagram, dtype=float)
    power_weights = np.abs(diagram[:, 1] - diagram[:, 0]) ** power
    total = np.sum(power_weights)
    if total == 0:
        weights = np.full(len(power_weights), 1.0 / len(power_weights))
    else:
        weights = power_weights / total
    silhouette_y = np.sum(tenting_values * weights[:, None], axis=0)
    return np.column_stack([t_values, silhouette_y])


def compute_mean_summary(summary_list: list[np.ndarray]) -> np.ndarray:
    """Mean of ``(t, value)`` curves after interpolating onto a common grid.

    The grid spans the union of all t-ranges; curves are zero-padded outside
    their own support.
    """
    if len(summary_list) == 0:
        raise ValueError("Cannot average an empty list of summaries.")
    all_x = np.concatenate([s[:, 0] for s in summary_list])
    x_min, x_max = all_x.min(), all_x.max()
    n_points = max(len(s) for s in summary_list)
    common_x = np.linspace(x_min, x_max, n_points)

    interpolated = np.array(
        [np.interp(common_x, s[:, 0], s[:, 1], left=0.0, right=0.0) for s in summary_list]
    )
    return np.column_stack([common_x, interpolated.mean(axis=0)])


@dataclass
class Summaries:
    """Summary curves for a single diagram."""

    landscape: np.ndarray
    silhouette: np.ndarray
    transformed_diagram: np.ndarray
    t_values: np.ndarray
    tenting_values: np.ndarray


def compute_summaries(
    finite_diagram: np.ndarray,
    *,
    transform: DiagramTransform | None = None,
    homology_dim: int = 0,
    resolution: int = 1000,
    silhouette_power: float = 3.0,
    landscape_order: int = 0,
) -> Summaries:
    """Transform ``finite_diagram`` and build its landscape and silhouette."""
    if transform is None:
        transform = RotateScale()
    finite_diagram = np.asarray(finite_diagram, dtype=float).reshape(-1, 2)
    transformed = transform(finite_diagram, homology_dim)
    tenting_values, t_values = compute_tenting_values(transformed, resolution)
    landscape = compute_landscape(tenting_values, t_values, order=landscape_order)
    silhouette = compute_silhouette(
        tenting_values, t_values, finite_diagram, power=silhouette_power
    )
    return Summaries(
        landscape=landscape,
        silhouette=silhouette,
        transformed_diagram=transformed,
        t_values=t_values,
        tenting_values=tenting_values,
    )
