"""The topological representation of an embedding set, and the transformer that builds it.

``TopologicalTransformer`` is the sklearn-style entry point: ``fit_transform``
takes an array of embeddings (or a distance matrix, or a prebuilt graph) and
returns a ``Representation`` holding the persistence diagram, the whole-graph
landscape/silhouette, and the bootstrap distribution of summaries with
confidence bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from .bootstrap import bootstrap_summaries
from .diagram_transforms import DiagramTransform, RotateScale
from .graph import build_graph
from .persistence import compute_persistence
from .summaries import compute_summaries

RandomState = int | np.random.Generator | None


@dataclass
class Representation:
    """Topological summary of one embedding set.

    The scoring layer compares two representations through their
    ``bootstrapped_*`` distributions (energy) or their ``mean_*`` curves
    (curve metrics).
    """

    homology_dim: int
    persistence_diagram: np.ndarray  # (n, 2), includes essential class
    finite_diagram: np.ndarray  # (m, 2), essential class removed
    transformed_diagram: np.ndarray  # whole-graph finite diagram, transformed
    full_landscape: np.ndarray  # (resolution, 2), whole graph
    full_silhouette: np.ndarray  # (resolution, 2), whole graph
    mean_landscape: np.ndarray  # (resolution, 2), bootstrap mean
    mean_silhouette: np.ndarray  # (resolution, 2), bootstrap mean
    bootstrapped_landscapes: np.ndarray  # (n_bootstrap, resolution, 2)
    bootstrapped_silhouettes: np.ndarray  # (n_bootstrap, resolution, 2)
    landscape_band: dict
    silhouette_band: dict
    skeleton1: np.ndarray
    sampled_node_indices: list
    normalization_factor: float | None
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def summary(self, kind: str, *, mean: bool = True) -> np.ndarray:
        """Return the ``landscape`` or ``silhouette`` curve (mean or whole-graph)."""
        if kind not in ("landscape", "silhouette"):
            raise ValueError(f"kind must be 'landscape' or 'silhouette', got {kind!r}.")
        prefix = "mean" if mean else "full"
        return getattr(self, f"{prefix}_{kind}")

    def bootstrapped(self, kind: str) -> np.ndarray:
        """Return the bootstrap replicate stack for ``landscape`` or ``silhouette``."""
        if kind not in ("landscape", "silhouette"):
            raise ValueError(f"kind must be 'landscape' or 'silhouette', got {kind!r}.")
        return getattr(self, f"bootstrapped_{kind}s")

    def __repr__(self) -> str:
        label = f" label={self.label!r}" if self.label else ""
        return (
            f"Representation(H{self.homology_dim}{label}, "
            f"diagram={self.persistence_diagram.shape}, "
            f"bootstrap={self.bootstrapped_landscapes.shape[0]})"
        )


class TopologicalTransformer:
    """Turn embeddings into a :class:`Representation`.

    Parameters
    ----------
    homology_dim:
        Homology dimension to summarize (0 for components, 1 for loops).
    theta:
        Rotation applied to the persistence diagram before summaries are built
        (shorthand for ``diagram_transform=RotateScale(theta=theta)``). This is
        the nonstandard knob of the method: ``-pi/4`` is the conventional
        rotation that flattens the diagonal; the default ``-3*pi/8`` adds the
        extra ``-pi/8`` tilt used in the original work. Mutually exclusive with
        ``diagram_transform``.
    diagram_transform:
        A full :class:`~persiscope.diagram_transforms.DiagramTransform` for
        anything beyond a rotation angle (custom alpha, custom schemes).
        Defaults to ``RotateScale()``.
    n_bootstrap, subsample:
        Number of subgraph draws and the fraction of nodes per draw.
    silhouette_power, landscape_order, tenting_resolution:
        Summary-function parameters.
    weight_method, normalize:
        Graph construction: distance metric and whether to min-max normalize
        edge weights (ignored when a prebuilt graph or distance matrix is given
        with ``normalize=False``).
    band_alpha:
        Significance level for the pointwise confidence bands (0.05 -> 95%).
    random_state:
        Seed or ``numpy`` generator for reproducible bootstrap draws.
    """

    def __init__(
        self,
        homology_dim: int = 0,
        theta: float | None = None,
        diagram_transform: DiagramTransform | None = None,
        n_bootstrap: int = 100,
        subsample: float = 0.8,
        silhouette_power: float = 0.5,
        landscape_order: int = 0,
        tenting_resolution: int = 1000,
        weight_method: str = "euclidean",
        normalize: bool = True,
        band_alpha: float = 0.05,
        random_state: RandomState = None,
        progress: bool = False,
    ):
        self.homology_dim = homology_dim
        if theta is not None and diagram_transform is not None:
            raise ValueError(
                "Pass either theta= (a rotation angle) or diagram_transform= "
                "(a full transform), not both."
            )
        self.theta = theta
        if diagram_transform is not None:
            self.diagram_transform = diagram_transform
        elif theta is not None:
            self.diagram_transform = RotateScale(theta=theta)
        else:
            self.diagram_transform = RotateScale()
        self.n_bootstrap = n_bootstrap
        self.subsample = subsample
        self.silhouette_power = silhouette_power
        self.landscape_order = landscape_order
        self.tenting_resolution = tenting_resolution
        self.weight_method = weight_method
        self.normalize = normalize
        self.band_alpha = band_alpha
        self.random_state = random_state
        self.progress = progress

    def fit_transform(
        self,
        embeddings: np.ndarray | None = None,
        *,
        distance_matrix: np.ndarray | None = None,
        graph: nx.Graph | None = None,
        label: str | None = None,
    ) -> Representation:
        """Build the :class:`Representation` for one input."""
        g, norm_factor = build_graph(
            embeddings,
            distance_matrix=distance_matrix,
            graph=graph,
            weight_method=self.weight_method,
            normalize=self.normalize,
        )

        # Whole-graph (deterministic) persistence + summaries.
        persistence = compute_persistence(g, homology_dim=self.homology_dim)
        full = compute_summaries(
            persistence.finite_diagram,
            transform=self.diagram_transform,
            homology_dim=self.homology_dim,
            resolution=self.tenting_resolution,
            silhouette_power=self.silhouette_power,
            landscape_order=self.landscape_order,
        )

        # Bootstrap distribution over subgraphs.
        rng = np.random.default_rng(self.random_state)
        boot = bootstrap_summaries(
            g,
            homology_dim=self.homology_dim,
            transform=self.diagram_transform,
            n_bootstrap=self.n_bootstrap,
            subsample=self.subsample,
            resolution=self.tenting_resolution,
            silhouette_power=self.silhouette_power,
            landscape_order=self.landscape_order,
            band_alpha=self.band_alpha,
            rng=rng,
            progress=self.progress,
        )

        return Representation(
            homology_dim=self.homology_dim,
            persistence_diagram=persistence.diagram,
            finite_diagram=persistence.finite_diagram,
            transformed_diagram=full.transformed_diagram,
            full_landscape=full.landscape,
            full_silhouette=full.silhouette,
            mean_landscape=boot.mean_landscape,
            mean_silhouette=boot.mean_silhouette,
            bootstrapped_landscapes=boot.bootstrapped_landscapes,
            bootstrapped_silhouettes=boot.bootstrapped_silhouettes,
            landscape_band=boot.landscape_band,
            silhouette_band=boot.silhouette_band,
            skeleton1=persistence.skeleton1,
            sampled_node_indices=boot.sampled_node_indices,
            normalization_factor=norm_factor,
            params={
                "homology_dim": self.homology_dim,
                "diagram_transform": repr(self.diagram_transform),
                "n_bootstrap": self.n_bootstrap,
                "subsample": self.subsample,
                "silhouette_power": self.silhouette_power,
                "landscape_order": self.landscape_order,
                "tenting_resolution": self.tenting_resolution,
                "weight_method": self.weight_method,
            },
            label=label,
        )


def transform(embeddings: np.ndarray | None = None, **kwargs) -> Representation:
    """Functional wrapper over :class:`TopologicalTransformer`.

    Transformer keyword arguments and ``fit_transform`` inputs
    (``distance_matrix=``, ``graph=``, ``label=``) may be mixed freely.
    """
    fit_keys = {"distance_matrix", "graph", "label"}
    fit_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in fit_keys}
    return TopologicalTransformer(**kwargs).fit_transform(embeddings, **fit_kwargs)
