"""Main entry point: compare a list of embedding sets pairwise.

``compare`` takes a list of embedding arrays and returns an all-pairs
:class:`~persiscope.scoring.results.ComparisonResult`. Each input is fitted
into a :class:`~persiscope.representation.Representation` exactly once, then
every pair is scored, so the (potentially expensive) topological transforms are
never recomputed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .diagram_transforms import DiagramTransform
from .representation import TopologicalTransformer
from .scoring.results import ComparisonResult
from .scoring.scorer import Scorer

RandomState = int | np.random.Generator | None


def compare(
    embeddings: Sequence[np.ndarray],
    *,
    method: str = "energy",
    summary: str = "landscape",
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
    run_pvalue: bool = False,
    n_permutations: int = 100,
    input_kind: str = "embeddings",
    labels: list[Any] | None = None,
    random_state: RandomState = None,
    progress: bool = False,
) -> ComparisonResult:
    """Fit each embedding set once, then score every pair.

    Parameters
    ----------
    embeddings:
        A list of ``k`` inputs. By default each is an embedding array of shape
        ``(n_i, d)``; set ``input_kind`` to ``"distance_matrix"`` or ``"graph"``
        to pass precomputed square distance matrices or ``networkx`` graphs.
    method, summary:
        Passed to :class:`~persiscope.scoring.scorer.Scorer` (e.g. ``"energy"``
        / ``"landscape"``).
    theta:
        Diagram rotation angle (the method's nonstandard knob) — shorthand for
        ``diagram_transform=RotateScale(theta=theta)``. Default ``-3*pi/8``;
        ``-pi/4`` is the conventional diagonal-flattening rotation. Mutually
        exclusive with ``diagram_transform``.
    labels:
        Optional names for the ``k`` inputs; become the matrix axes.
    random_state:
        Seed or generator. Each input gets an independent child stream so the
        run is reproducible without correlating the bootstrap draws.

    Returns
    -------
    ComparisonResult
        ``.matrix`` (k x k), ``.pvalues`` (or ``None``), ``.representations``,
        ``.labels``, and ``.to_frame()``.
    """
    inputs = list(embeddings)
    k = len(inputs)
    if k < 2:
        raise ValueError("compare() needs at least 2 embedding sets.")
    if input_kind not in ("embeddings", "distance_matrix", "graph"):
        raise ValueError(
            f"input_kind must be 'embeddings', 'distance_matrix', or 'graph', got {input_kind!r}."
        )
    if labels is not None and len(labels) != k:
        raise ValueError(f"Got {len(labels)} labels for {k} inputs.")

    transformer = TopologicalTransformer(
        homology_dim=homology_dim,
        theta=theta,
        diagram_transform=diagram_transform,
        n_bootstrap=n_bootstrap,
        subsample=subsample,
        silhouette_power=silhouette_power,
        landscape_order=landscape_order,
        tenting_resolution=tenting_resolution,
        weight_method=weight_method,
        normalize=normalize,
        band_alpha=band_alpha,
        progress=progress,
    )

    # Independent, reproducible per-input bootstrap streams.
    seeds = np.random.SeedSequence(_entropy(random_state)).spawn(k)

    representations = []
    fit_iter = enumerate(inputs)
    if progress:
        from tqdm.auto import tqdm

        fit_iter = tqdm(list(fit_iter), desc="fitting representations")

    for i, item in fit_iter:
        transformer.random_state = np.random.default_rng(seeds[i])
        label = labels[i] if labels is not None else None
        kwargs = {"label": label}
        if input_kind == "embeddings":
            rep = transformer.fit_transform(item, **kwargs)
        elif input_kind == "distance_matrix":
            rep = transformer.fit_transform(distance_matrix=item, **kwargs)
        else:
            rep = transformer.fit_transform(graph=item, **kwargs)
        representations.append(rep)

    scorer = Scorer(
        method=method,
        summary=summary,
        run_pvalue=run_pvalue,
        n_permutations=n_permutations,
        random_state=np.random.default_rng(np.random.SeedSequence(_entropy(random_state))),
    )
    score_result = scorer.score_all(representations, labels=labels)

    return ComparisonResult(
        matrix=score_result.matrix,
        method=method,
        summary=summary,
        pvalues=score_result.pvalues,
        labels=score_result.labels,
        representations=representations,
    )


def _entropy(random_state: RandomState):
    """Coerce a seed/generator/None into entropy for a SeedSequence."""
    if random_state is None or isinstance(random_state, int):
        return random_state
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**63 - 1))
    return random_state
