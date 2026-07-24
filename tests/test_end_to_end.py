import numpy as np

import persiscope as ps


def _two_clusters(rng, n, dim, sep):
    half = n // 2
    a = rng.normal(-sep, 1, size=(half, dim))
    b = rng.normal(sep, 1, size=(n - half, dim))
    return np.vstack([a, b])


def test_compare_separates_dissimilar_topology():
    """Two single blobs should look alike; a two-cluster set should look different."""
    rng = np.random.default_rng(42)
    dim = 6
    blob_a = rng.normal(0, 1, size=(40, dim))
    blob_b = rng.normal(0, 1, size=(40, dim))
    clusters = _two_clusters(rng, 40, dim, sep=6.0)

    result = ps.compare(
        [blob_a, blob_b, clusters],
        method="energy",
        summary="landscape",
        homology_dim=0,
        n_bootstrap=30,
        labels=["blob_a", "blob_b", "clusters"],
        random_state=0,
    )

    m = result.matrix
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 0.0)
    # the two blobs are more alike than either is to the clustered set
    assert m[0, 1] < m[0, 2]
    assert m[0, 1] < m[1, 2]


def test_compare_returns_reusable_representations():
    rng = np.random.default_rng(1)
    embs = [rng.normal(size=(20, 4)) for _ in range(3)]
    result = ps.compare(embs, method="euclidean", n_bootstrap=10, random_state=0)
    assert len(result.representations) == 3
    assert all(isinstance(r, ps.Representation) for r in result.representations)


def test_compare_with_pvalues():
    rng = np.random.default_rng(2)
    embs = [rng.normal(size=(20, 4)) for _ in range(3)]
    result = ps.compare(
        embs, method="energy", n_bootstrap=10, run_pvalue=True,
        n_permutations=20, random_state=0,
    )
    assert result.pvalues is not None
    assert result.pvalues.shape == (3, 3)
    offdiag = result.pvalues[~np.eye(3, dtype=bool)]
    assert np.all((offdiag > 0) & (offdiag <= 1))


def test_compare_reproducible():
    rng = np.random.default_rng(5)
    embs = [rng.normal(size=(18, 4)) for _ in range(3)]
    r1 = ps.compare(embs, method="energy", n_bootstrap=12, random_state=123)
    r2 = ps.compare(embs, method="energy", n_bootstrap=12, random_state=123)
    assert np.allclose(r1.matrix, r2.matrix)
