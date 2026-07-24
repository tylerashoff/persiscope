"""Vietoris-Rips persistence from a weighted graph.

The graph's edge ``weight`` attributes are read as a distance matrix, a Rips
complex is built with GUDHI, and persistence intervals are extracted for the
requested homology dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from gudhi import RipsComplex


def _distance_matrix_from_graph(graph: nx.Graph) -> np.ndarray:
    """Dense distance matrix from a graph's ``weight`` edge attributes.

    Node labels may be arbitrary (subgraphs keep their parent labels), so rows
    and columns follow ``list(graph.nodes())`` order.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    index = {node: k for k, node in enumerate(nodes)}
    dist = np.zeros((n, n), dtype=float)
    for u, v, data in graph.edges(data=True):
        w = data.get("weight")
        if w is None:
            raise ValueError(f"Edge ({u}, {v}) has no 'weight' attribute.")
        i, j = index[u], index[v]
        dist[i, j] = w
        dist[j, i] = w
    return dist


def build_simplex_tree(graph: nx.Graph, homology_dim: int):
    """Rips complex simplex tree for ``graph`` up to dimension ``homology_dim + 1``."""
    dist = _distance_matrix_from_graph(graph)
    rips = RipsComplex(distance_matrix=np.tril(dist), max_edge_length=dist.max())
    # +1 because homology dimension is 0-indexed and we need one dimension above.
    simplex_tree = rips.create_simplex_tree(max_dimension=homology_dim + 1)
    return simplex_tree


@dataclass
class Persistence:
    """Raw persistence output for one homology dimension.

    Attributes
    ----------
    diagram:
        ``(n_points, 2)`` array of ``(birth, death)`` pairs, including any
        essential (infinite-death) class.
    finite_diagram:
        ``diagram`` with non-finite deaths removed — the points used to build
        landscapes and silhouettes.
    skeleton1:
        ``(n_edges, 3)`` array of ``[node_i, node_j, filtration]`` for 1-simplices.
    homology_dim:
        The homology dimension these intervals were computed in.
    """

    diagram: np.ndarray
    finite_diagram: np.ndarray
    skeleton1: np.ndarray
    homology_dim: int
    simplex_tree: object = field(repr=False, default=None)


def compute_persistence(graph: nx.Graph, homology_dim: int = 0) -> Persistence:
    """Compute Rips persistence for ``graph`` in ``homology_dim``.

    For ``homology_dim == 0`` a negative ``min_persistence`` is used so that
    zero-persistence merge events (from zero-weight edges) are retained — this
    keeps one diagram point per node, matching the merging structure of the
    connected components.
    """
    simplex_tree = build_simplex_tree(graph, homology_dim)

    min_persistence = -1 if homology_dim == 0 else 0
    simplex_tree.compute_persistence(min_persistence=min_persistence)

    diagram = np.array(simplex_tree.persistence_intervals_in_dimension(homology_dim))
    if diagram.size == 0:
        diagram = np.zeros((0, 2))

    skeleton1: list[list] = [
        [*simplex, filtration]
        for simplex, filtration in simplex_tree.get_skeleton(1)
        if len(simplex) == 2
    ]
    skeleton1_arr = np.array(skeleton1) if skeleton1 else np.zeros((0, 3))

    finite = diagram[np.isfinite(diagram).all(axis=1)] if len(diagram) else diagram

    return Persistence(
        diagram=diagram,
        finite_diagram=finite,
        skeleton1=skeleton1_arr,
        homology_dim=homology_dim,
        simplex_tree=simplex_tree,
    )
