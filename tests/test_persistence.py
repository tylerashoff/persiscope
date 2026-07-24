import numpy as np
import pytest

from persiscope.graph import build_graph
from persiscope.persistence import compute_persistence


@pytest.fixture
def small_graph():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(12, 5))
    graph, _ = build_graph(emb, normalize=False)
    return graph, 12


def test_h0_diagram_has_one_point_per_node(small_graph):
    graph, n = small_graph
    pers = compute_persistence(graph, homology_dim=0)
    # H0: one interval per component birth => one per node.
    assert pers.diagram.shape == (n, 2)


def test_finite_diagram_drops_essential_class(small_graph):
    graph, n = small_graph
    pers = compute_persistence(graph, homology_dim=0)
    assert np.isfinite(pers.finite_diagram).all()
    # Exactly one essential (infinite-death) H0 class.
    assert len(pers.finite_diagram) == n - 1


def test_skeleton1_shape(small_graph):
    graph, n = small_graph
    pers = compute_persistence(graph, homology_dim=0)
    assert pers.skeleton1.shape == (n * (n - 1) // 2, 3)


def test_homology_dim_recorded(small_graph):
    graph, _ = small_graph
    pers = compute_persistence(graph, homology_dim=0)
    assert pers.homology_dim == 0
    assert pers.simplex_tree is not None
