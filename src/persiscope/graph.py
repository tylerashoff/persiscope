"""Build a weighted graph from embeddings, a distance matrix, or a prebuilt graph.

The topological pipeline operates on a ``networkx`` graph whose edge ``weight``
attributes are pairwise distances. This module is the single entry point for
turning user input into that graph, mirroring the fully-connected construction
used in the original research pipeline while adding distance-matrix and
prebuilt-graph inputs.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

WeightMethod = str  # "euclidean" | "cosine" | "manhattan"


def compute_weight(vec1: np.ndarray, vec2: np.ndarray, method: WeightMethod) -> float:
    """Distance between two vectors under ``method``."""
    if method == "euclidean":
        return float(np.linalg.norm(vec1 - vec2))
    if method == "cosine":
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(1.0 - np.dot(vec1, vec2) / (norm1 * norm2))
    if method == "manhattan":
        return float(np.sum(np.abs(vec1 - vec2)))
    raise ValueError(
        f"Unknown weight method: {method!r}. Supported: euclidean, cosine, manhattan."
    )


def _distance_matrix(embeddings: np.ndarray, method: WeightMethod) -> np.ndarray:
    """Full pairwise distance matrix for a stack of row vectors."""
    n = len(embeddings)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_weight(embeddings[i], embeddings[j], method)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _graph_from_distance_matrix(
    dist: np.ndarray,
    *,
    normalize: bool = True,
    normalization_factor: float | None = None,
    embeddings: np.ndarray | None = None,
) -> tuple[nx.Graph, float | None]:
    """Fully-connected graph from a square distance matrix.

    Weights are optionally min-max normalized. The divisor used for
    normalization is returned so downstream code can recover the original
    scale.
    """
    dist = np.asarray(dist, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape {dist.shape}.")
    n = dist.shape[0]

    graph = nx.Graph()
    for i in range(n):
        node_vec = embeddings[i] if embeddings is not None else None
        graph.add_node(i, vector=node_vec)

    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(dist[i, j]), index1=i, index2=j)

    if normalize and graph.number_of_edges() > 0:
        weights = [graph[u][v]["weight"] for u, v in graph.edges()]
        min_w, max_w = min(weights), max(weights)
        if normalization_factor is None:
            normalization_factor = max_w - min_w
        if max_w > min_w:
            for u, v in graph.edges():
                graph[u][v]["weight"] = (graph[u][v]["weight"] - min_w) / normalization_factor

    return graph, normalization_factor


def build_graph(
    embeddings: np.ndarray | None = None,
    *,
    distance_matrix: np.ndarray | None = None,
    graph: nx.Graph | None = None,
    weight_method: WeightMethod = "euclidean",
    normalize: bool = True,
    normalization_factor: float | None = None,
) -> tuple[nx.Graph, float | None]:
    """Return ``(graph, normalization_factor)`` from one of three inputs.

    Exactly one of ``embeddings``, ``distance_matrix``, or ``graph`` must be
    given.

    - ``embeddings``: array ``(n_points, n_features)`` -> fully-connected graph
      whose edge weights are pairwise ``weight_method`` distances.
    - ``distance_matrix``: square ``(n, n)`` array of precomputed distances.
    - ``graph``: a prebuilt ``networkx.Graph`` with ``weight`` edge attributes,
      returned unchanged (``normalization_factor`` is unknown -> ``None``).
    """
    provided = [x is not None for x in (embeddings, distance_matrix, graph)]
    if sum(provided) != 1:
        raise ValueError(
            "Provide exactly one of embeddings=, distance_matrix=, or graph=."
        )

    if graph is not None:
        return graph, None

    if distance_matrix is not None:
        return _graph_from_distance_matrix(
            distance_matrix,
            normalize=normalize,
            normalization_factor=normalization_factor,
        )

    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (n_points, n_features), got shape {embeddings.shape}."
        )
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 points to build a graph.")

    dist = _distance_matrix(embeddings, weight_method)
    return _graph_from_distance_matrix(
        dist,
        normalize=normalize,
        normalization_factor=normalization_factor,
        embeddings=embeddings,
    )
