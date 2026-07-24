import networkx as nx
import numpy as np
import pytest

from persiscope.graph import build_graph, compute_weight


def test_build_graph_from_embeddings_is_fully_connected():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(10, 4))
    graph, norm = build_graph(emb, normalize=True)

    assert graph.number_of_nodes() == 10
    assert graph.number_of_edges() == 10 * 9 // 2
    weights = [d["weight"] for _, _, d in graph.edges(data=True)]
    assert min(weights) >= 0.0 and max(weights) <= 1.0 + 1e-9
    # normalization factor is the raw (max - min) distance spread
    assert norm > 0


def test_build_graph_weights_symmetric_and_match_metric():
    emb = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    graph, _ = build_graph(emb, normalize=False)
    # euclidean distance 0->1 is 5, 0->2 is 10, 1->2 is 5
    assert graph[0][1]["weight"] == pytest.approx(5.0)
    assert graph[0][2]["weight"] == pytest.approx(10.0)
    assert graph[1][0]["weight"] == graph[0][1]["weight"]


def test_build_graph_from_distance_matrix():
    dist = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]])
    graph, _norm = build_graph(distance_matrix=dist, normalize=False)
    assert graph.number_of_nodes() == 3
    assert graph[1][2]["weight"] == pytest.approx(6.0)


def test_build_graph_passthrough():
    g = nx.Graph()
    g.add_edge(0, 1, weight=0.5)
    out, norm = build_graph(graph=g)
    assert out is g
    assert norm is None


def test_build_graph_requires_exactly_one_input():
    emb = np.zeros((3, 2))
    with pytest.raises(ValueError):
        build_graph()
    with pytest.raises(ValueError):
        build_graph(emb, distance_matrix=np.zeros((3, 3)))


def test_build_graph_rejects_single_point():
    with pytest.raises(ValueError):
        build_graph(np.zeros((1, 2)))


@pytest.mark.parametrize("method,expected", [("euclidean", 5.0), ("manhattan", 7.0)])
def test_compute_weight(method, expected):
    v1 = np.array([0.0, 0.0])
    v2 = np.array([3.0, 4.0])
    assert compute_weight(v1, v2, method) == pytest.approx(expected)


def test_compute_weight_cosine_orthogonal():
    assert compute_weight(np.array([1.0, 0.0]), np.array([0.0, 1.0]), "cosine") == pytest.approx(1.0)
