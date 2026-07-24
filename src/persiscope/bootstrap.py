"""Subsampling bootstrap over subgraphs.

Repeatedly draws random node subsets, recomputes persistence summaries on each
induced subgraph, and aggregates them into mean curves with confidence bands.
This is what makes the resulting representation stable to sampling noise and
gives the scoring layer a distribution to compare, rather than a single curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from .bands import pointwise_band
from .diagram_transforms import DiagramTransform, RotateScale
from .persistence import compute_persistence
from .summaries import compute_mean_summary, compute_summaries


@dataclass
class BootstrapResult:
    """Aggregated bootstrap output for one representation."""

    bootstrapped_landscapes: np.ndarray  # (n_bootstrap, resolution, 2)
    bootstrapped_silhouettes: np.ndarray  # (n_bootstrap, resolution, 2)
    mean_landscape: np.ndarray  # (resolution, 2)
    mean_silhouette: np.ndarray  # (resolution, 2)
    landscape_band: dict
    silhouette_band: dict
    sampled_node_indices: list[list]


def bootstrap_summaries(
    graph: nx.Graph,
    *,
    homology_dim: int = 0,
    transform: DiagramTransform | None = None,
    n_bootstrap: int = 100,
    subsample: float = 0.8,
    resolution: int = 1000,
    silhouette_power: float = 0.5,
    landscape_order: int = 0,
    band_alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    progress: bool = False,
) -> BootstrapResult:
    """Run the subsampling bootstrap and aggregate summary curves."""
    if transform is None:
        transform = RotateScale()
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(graph.nodes())
    if len(nodes) == 0:
        raise ValueError("Cannot bootstrap an empty graph.")
    sample_size = max(2, int(len(nodes) * subsample))

    landscapes: list[np.ndarray] = []
    silhouettes: list[np.ndarray] = []
    sampled_indices: list[list] = []

    iterator = range(n_bootstrap)
    if progress:
        iterator = tqdm(iterator, desc="bootstrap", leave=False)

    for _ in iterator:
        chosen = rng.choice(len(nodes), size=sample_size, replace=False)
        sampled_nodes = [nodes[k] for k in chosen]
        subgraph = graph.subgraph(sampled_nodes).copy()

        persistence = compute_persistence(subgraph, homology_dim=homology_dim)
        summaries = compute_summaries(
            persistence.finite_diagram,
            transform=transform,
            homology_dim=homology_dim,
            resolution=resolution,
            silhouette_power=silhouette_power,
            landscape_order=landscape_order,
        )
        landscapes.append(summaries.landscape)
        silhouettes.append(summaries.silhouette)
        sampled_indices.append(sampled_nodes)

    if len(landscapes) == 0:
        raise ValueError("No bootstrap samples were produced.")

    mean_landscape = compute_mean_summary(landscapes)
    mean_silhouette = compute_mean_summary(silhouettes)
    landscape_band = pointwise_band(landscapes, mean_landscape, alpha=band_alpha)
    silhouette_band = pointwise_band(silhouettes, mean_silhouette, alpha=band_alpha)

    return BootstrapResult(
        bootstrapped_landscapes=np.array(landscapes),
        bootstrapped_silhouettes=np.array(silhouettes),
        mean_landscape=mean_landscape,
        mean_silhouette=mean_silhouette,
        landscape_band=landscape_band,
        silhouette_band=silhouette_band,
        sampled_node_indices=sampled_indices,
    )
