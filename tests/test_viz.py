import numpy as np
import pytest

import persiscope as ps

matplotlib = pytest.importorskip("matplotlib")
plotly = pytest.importorskip("plotly")
pytestmark = pytest.mark.viz


@pytest.fixture(scope="module")
def comparison():
    rng = np.random.default_rng(0)
    embs = [rng.normal(size=(20, 4)) for _ in range(3)]
    return ps.compare(embs, method="energy", n_bootstrap=8, labels=["x", "y", "z"], random_state=0)


def test_plot_persistence_diagram(comparison):
    import matplotlib

    matplotlib.use("Agg")
    fig = ps.viz.plot_persistence_diagram(comparison.representations[0])
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_landscape_and_silhouette(comparison):
    import matplotlib

    matplotlib.use("Agg")
    rep = comparison.representations[0]
    assert ps.viz.plot_landscape(rep) is not None
    assert ps.viz.plot_silhouette(rep) is not None


def test_plot_score_heatmap(comparison):
    fig = ps.viz.plot_score_heatmap(comparison)
    # a plotly Figure with one heatmap trace
    assert fig.data[0].type == "heatmap"
    assert list(fig.data[0].x) == ["x", "y", "z"]


def test_plot_comparison_report(comparison):
    import matplotlib

    matplotlib.use("Agg")
    fig = ps.viz.plot_comparison_report(
        comparison.representations, summary="landscape",
        n_permutations=20, random_state=0,
    )
    # 3 top panels + 3 pair histograms + 2 heatmaps + 2 colorbars
    assert len(fig.axes) == 3 + 3 + 2 + 2


def test_plot_comparison_report_needs_two(comparison):
    with pytest.raises(ValueError):
        ps.viz.plot_comparison_report(comparison.representations[:1])


def test_plot_baseline_report(comparison):
    import matplotlib

    matplotlib.use("Agg")
    fig = ps.viz.plot_baseline_report(
        comparison.representations, baseline=0,
        n_permutations=20, random_state=0,
    )
    # 2 summary panels + energy panel + 2 heatmap strips + 1 colorbar
    assert len(fig.axes) == 2 + 1 + 2 + 1


def test_plot_baseline_report_bad_baseline(comparison):
    with pytest.raises(ValueError):
        ps.viz.plot_baseline_report(comparison.representations, baseline=9)
