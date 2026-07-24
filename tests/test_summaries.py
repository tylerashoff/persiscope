import numpy as np
import pytest

from persiscope.diagram_transforms import Identity
from persiscope.summaries import (
    compute_landscape,
    compute_mean_summary,
    compute_silhouette,
    compute_summaries,
    compute_tenting_values,
)


def test_tenting_peaks_at_x_with_height_y():
    diagram = np.array([[1.0, 0.5]])  # tent from 0.5 to 1.5, peak 0.5 at t=1.0
    tenting, t = compute_tenting_values(diagram, resolution=1001)
    assert tenting.shape == (1, 1001)
    peak_idx = np.argmax(tenting[0])
    assert t[peak_idx] == pytest.approx(1.0, abs=2e-3)
    assert tenting[0].max() == pytest.approx(0.5, abs=1e-3)


def test_landscape_orders_pick_kth_largest():
    diagram = np.array([[1.0, 0.6], [1.0, 0.3]])  # two tents peaking at t=1
    tenting, t = compute_tenting_values(diagram, resolution=2001)
    l0 = compute_landscape(tenting, t, order=0)
    l1 = compute_landscape(tenting, t, order=1)
    assert l0[:, 1].max() == pytest.approx(0.6, abs=1e-3)
    assert l1[:, 1].max() == pytest.approx(0.3, abs=1e-3)
    # first landscape dominates the second everywhere
    assert np.all(l0[:, 1] + 1e-9 >= l1[:, 1])


def test_silhouette_equal_persistence_is_tent_average():
    # equal |col1 - col0| -> equal silhouette weights -> silhouette is the tent mean
    diagram = np.array([[1.0, 1.5], [3.0, 3.5]])
    tenting, t = compute_tenting_values(diagram, resolution=1001)
    sil = compute_silhouette(tenting, t, diagram, power=3.0)
    expected = tenting.mean(axis=0)
    assert np.allclose(sil[:, 1], expected)


def test_mean_summary_of_identical_curves():
    curve = np.column_stack([np.linspace(0, 1, 50), np.sin(np.linspace(0, 1, 50))])
    mean = compute_mean_summary([curve, curve, curve])
    assert np.allclose(mean[:, 1], curve[:, 1], atol=1e-6)


def test_compute_summaries_shapes():
    rng = np.random.default_rng(3)
    diagram = np.abs(rng.normal(size=(6, 2)))
    diagram[:, 1] += diagram[:, 0]  # ensure death > birth
    out = compute_summaries(diagram, transform=Identity(), resolution=200)
    assert out.landscape.shape == (200, 2)
    assert out.silhouette.shape == (200, 2)
    assert out.tenting_values.shape == (6, 200)
