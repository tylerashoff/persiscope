import numpy as np
import pytest

from persiscope.bands import pointwise_band


@pytest.fixture
def bootstrap_curves():
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 60)
    base = np.sin(np.pi * t)
    curves = [np.column_stack([t, base + rng.normal(0, 0.05, size=t.size)]) for _ in range(40)]
    mean = np.column_stack([t, np.mean([c[:, 1] for c in curves], axis=0)])
    return curves, mean


def test_band_shapes_and_ordering(bootstrap_curves):
    curves, mean = bootstrap_curves
    band = pointwise_band(curves, mean)
    n = len(mean)
    assert set(band) == {"lower_bound", "upper_bound", "half_bound"}
    for key in ("lower_bound", "upper_bound", "half_bound"):
        assert band[key].shape == (n,)
    assert np.all(band["half_bound"] >= -1e-12)
    assert np.all(band["upper_bound"] >= band["lower_bound"] - 1e-9)


def test_band_covers_mean(bootstrap_curves):
    curves, mean = bootstrap_curves
    band = pointwise_band(curves, mean, alpha=0.05)
    # the mean of the bootstrap distribution sits inside its own 95% band
    assert np.all(mean[:, 1] >= band["lower_bound"] - 1e-9)
    assert np.all(mean[:, 1] <= band["upper_bound"] + 1e-9)


def test_tighter_alpha_widens_band(bootstrap_curves):
    curves, mean = bootstrap_curves
    band95 = pointwise_band(curves, mean, alpha=0.05)
    band50 = pointwise_band(curves, mean, alpha=0.50)
    assert band95["half_bound"].mean() > band50["half_bound"].mean()


def test_empty_bootstrap_raises():
    with pytest.raises(ValueError):
        pointwise_band([], np.zeros((5, 2)))
