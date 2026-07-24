import numpy as np
import pytest

from persiscope import Scorer, TopologicalTransformer, score_matrix
from persiscope.scoring import energy_statistic
from persiscope.scoring.permutation import permutation_pvalue


@pytest.fixture(scope="module")
def reps():
    rng = np.random.default_rng(11)
    tf = TopologicalTransformer(homology_dim=0, n_bootstrap=12, random_state=0)
    a = tf.fit_transform(rng.normal(size=(30, 6)), label="a")
    b = tf.fit_transform(rng.normal(size=(30, 6)), label="b")
    c = tf.fit_transform(rng.normal(4, 1, size=(30, 6)), label="c")
    return a, b, c


def test_energy_self_is_near_zero(reps):
    a, _, _ = reps
    s = Scorer(method="energy", summary="landscape").score(a, a)
    assert s.score == pytest.approx(0.0, abs=1e-9)


def test_energy_symmetric(reps):
    a, b, _ = reps
    scorer = Scorer(method="energy", summary="landscape")
    assert scorer.score(a, b).score == pytest.approx(scorer.score(b, a).score)


def test_unknown_method_rejected(reps):
    a, b, _ = reps
    with pytest.raises(ValueError):
        Scorer(method="morse").score(a, b)


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "js", "wasserstein", "spectral"])
def test_curve_metrics_run(reps, metric):
    a, b, _ = reps
    s = Scorer(method=metric, summary="landscape").score(a, b)
    assert np.isfinite(s.score)
    assert s.method == metric


def test_permutation_pvalue_bounds(reps):
    a, b, _ = reps
    s = Scorer(method="energy", run_pvalue=True, n_permutations=50, random_state=0).score(a, b)
    assert s.perm_scores.shape == (50,)
    assert 1 / 51 <= s.pvalue <= 1.0


def test_permutation_direct_helper():
    rng = np.random.default_rng(0)
    s1 = rng.normal(0, 1, size=(8, 20, 2))
    s2 = rng.normal(0, 1, size=(8, 20, 2))
    observed = energy_statistic(s1, s2)
    perm, p = permutation_pvalue(observed, energy_statistic, s1, s2, n_permutations=30, rng=rng)
    assert len(perm) == 30
    assert 0 < p <= 1


def test_score_matrix_symmetric_zero_diagonal(reps):
    a, b, c = reps
    sm = score_matrix([a, b, c], method="energy", labels=["a", "b", "c"])
    assert sm.matrix.shape == (3, 3)
    assert np.allclose(np.diag(sm.matrix), 0.0)
    assert np.allclose(sm.matrix, sm.matrix.T)
    assert sm.pvalues is None


def test_score_matrix_to_frame(reps):
    a, b, c = reps
    sm = score_matrix([a, b, c], method="euclidean", labels=["a", "b", "c"])
    df = sm.to_frame()
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (3, 3)


def test_scorer_rejects_bad_summary():
    with pytest.raises(ValueError):
        Scorer(method="energy", summary="bogus")
