"""One-figure comparison report across a set of representations.

Layout (matplotlib):

- **Top row** — each representation's mean summary curve with its confidence band.
- **Middle row** — one panel per pair: the energy-statistic permutation
  distribution with the observed score and p-value.
- **Bottom row** — annotated heatmaps of curve distances (JS and Wasserstein by
  default), each cell showing the distance with its permutation p-value.

Intended for small model sets (the middle row grows as C(k, 2))."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from ..scoring.scorer import Scorer
from ._backends import require_matplotlib

RandomState = int | np.random.Generator | None


def _rep_labels(representations: Sequence, labels: list[str] | None) -> list[str]:
    if labels is not None:
        return [str(v) for v in labels]
    return [str(getattr(r, "label", None) or i) for i, r in enumerate(representations)]


def plot_comparison_report(
    representations: Sequence,
    *,
    summary: str = "landscape",
    curve_metrics: Sequence[str] = ("js", "wasserstein"),
    n_permutations: int = 200,
    labels: list[str] | None = None,
    random_state: RandomState = None,
    figsize: tuple | None = None,
):
    """Build the three-row comparison report and return the matplotlib ``Figure``.

    Parameters
    ----------
    representations:
        Fitted :class:`~persiscope.representation.Representation` objects
        (e.g. ``compare(...).representations``). Works best for 2-5 models.
    summary:
        ``"landscape"`` or ``"silhouette"`` — used for every panel.
    curve_metrics:
        Which curve distances fill the bottom row (one heatmap each).
    n_permutations:
        Permutations for every p-value in the figure.
    """
    plt = require_matplotlib()

    reps = list(representations)
    k = len(reps)
    if k < 2:
        raise ValueError("Need at least 2 representations for a comparison report.")
    names = _rep_labels(reps, labels)
    pairs = list(combinations(range(k), 2))

    # --- scores ------------------------------------------------------------
    energy = Scorer(
        method="energy", summary=summary, run_pvalue=True,
        n_permutations=n_permutations, random_state=random_state,
    )
    energy_results = {pair: energy.score(reps[pair[0]], reps[pair[1]]) for pair in pairs}

    metric_matrices = {}
    for metric in curve_metrics:
        scorer = Scorer(
            method=metric, summary=summary, run_pvalue=True,
            n_permutations=n_permutations, random_state=random_state,
        )
        scores = np.zeros((k, k))
        pvals = np.full((k, k), np.nan)
        for i, j in pairs:
            res = scorer.score(reps[i], reps[j])
            scores[i, j] = scores[j, i] = res.score
            pvals[i, j] = pvals[j, i] = res.pvalue
        metric_matrices[metric] = (scores, pvals)

    # --- figure ------------------------------------------------------------
    n_mid = len(pairs)
    n_bottom = len(curve_metrics)
    n_cols = max(k, n_mid, n_bottom, 1)
    if figsize is None:
        figsize = (3.2 * n_cols, 9.5)

    fig = plt.figure(figsize=figsize)
    rows = fig.add_gridspec(3, 1, hspace=0.6)
    top_grid = rows[0].subgridspec(1, k, wspace=0.25)
    mid_grid = rows[1].subgridspec(1, n_mid, wspace=0.3)
    bottom_grid = rows[2].subgridspec(1, n_bottom, wspace=0.45)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Top row: mean summaries with bands, shared y-limits for comparability.
    curves = [r.summary(summary) for r in reps]
    y_max = max(c[:, 1].max() for c in curves) * 1.1
    top_axes = []
    for i, rep in enumerate(reps):
        ax = fig.add_subplot(top_grid[0, i])
        curve = curves[i]
        color = colors[i % len(colors)]
        ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.6)
        band = getattr(rep, f"{summary}_band", None)
        if band is not None and len(band["lower_bound"]) == len(curve):
            ax.fill_between(
                curve[:, 0], band["lower_bound"], band["upper_bound"],
                color=color, alpha=0.2,
            )
        ax.set_ylim(0, y_max)
        ax.set_title(names[i], fontsize=10)
        ax.set_xlabel("t", fontsize=8)
        if i == 0:
            ax.set_ylabel(f"mean {summary}", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8)
        top_axes.append(ax)

    # Middle row: energy permutation distribution per pair.
    for col, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(mid_grid[0, col])
        res = energy_results[(i, j)]
        ax.hist(res.perm_scores, bins=20, color="0.75", edgecolor="0.55")
        ax.axvline(res.score, color="crimson", lw=1.8)
        ax.set_title(f"{names[i]} vs {names[j]}", fontsize=10)
        ax.set_xlabel("energy statistic", fontsize=8)
        if col == 0:
            ax.set_ylabel("permutations", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(
            0.97, 0.95, f"E = {res.score:.3g}\np = {res.pvalue:.3g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
        )

    # Bottom row: annotated curve-metric heatmaps (score + p-value per cell).
    for col, metric in enumerate(curve_metrics):
        ax = fig.add_subplot(bottom_grid[0, col])
        scores, pvals = metric_matrices[metric]
        im = ax.imshow(scores, cmap="viridis")
        ax.set_xticks(range(k), names, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(k), names, fontsize=8)
        ax.set_title(f"{metric} distance", fontsize=10)
        vmid = scores.max() / 2 if scores.max() > 0 else 0.5
        for i in range(k):
            for j in range(k):
                if i == j:
                    text = "—"
                else:
                    text = f"{scores[i, j]:.3g}\np={pvals[i, j]:.2g}"
                ax.text(
                    j, i, text, ha="center", va="center", fontsize=7.5,
                    color="white" if scores[i, j] < vmid else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"persiscope comparison report — {summary} (perms={n_permutations})",
        fontsize=12, y=0.99,
    )
    return fig


def _stars(pvalue: float | None) -> str:
    """Significance stars: *** <0.001, ** <0.01, * <0.05, else ns."""
    if pvalue is None or np.isnan(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def plot_baseline_report(
    representations: Sequence,
    *,
    baseline: int = 0,
    curve_metrics: Sequence[str] = ("wasserstein", "js"),
    n_permutations: int = 200,
    labels: list[str] | None = None,
    random_state: RandomState = None,
    figsize: tuple | None = None,
):
    """Compare every representation to a baseline, in one figure.

    Layout (top to bottom):

    1. Overlaid **mean landscapes** with confidence bands, one color per model.
    2. Overlaid **mean silhouettes** (shared x-axis with panel 1).
    3. **Energy statistic** vs. the baseline as a line per summary type
       (landscapes and silhouettes), with permutation-significance stars under
       each model label (order: landscapes, silhouettes).
    4. One 2-row heatmap strip per curve metric (rows: silhouettes,
       landscapes; columns: models): the distance of each model's mean curve
       from the baseline's, annotated with significance stars.

    Parameters
    ----------
    representations:
        Fitted representations; ``representations[baseline]`` is the reference
        every other model is compared to.
    curve_metrics:
        Curve distances for the bottom strips.
    n_permutations:
        Permutations behind every star in the figure.
    """
    plt = require_matplotlib()

    reps = list(representations)
    k = len(reps)
    if k < 2:
        raise ValueError("Need at least 2 representations for a baseline report.")
    if not 0 <= baseline < k:
        raise ValueError(f"baseline index {baseline} out of range for {k} representations.")
    names = _rep_labels(reps, labels)
    base = reps[baseline]
    summaries = ("landscape", "silhouette")

    # --- scores vs. baseline ------------------------------------------------
    def _score_vs_base(method: str, summary: str, rep_index: int):
        if rep_index == baseline:
            return 0.0, None  # self-comparison: zero by definition, no test
        scorer = Scorer(
            method=method, summary=summary, run_pvalue=True,
            n_permutations=n_permutations, random_state=random_state,
        )
        res = scorer.score(base, reps[rep_index])
        return res.score, res.pvalue

    energy = {
        summary: [_score_vs_base("energy", summary, i) for i in range(k)]
        for summary in summaries
    }
    metric_scores = {
        metric: {
            summary: [_score_vs_base(metric, summary, i) for i in range(k)]
            for summary in summaries
        }
        for metric in curve_metrics
    }

    # --- figure ---------------------------------------------------------------
    n_metrics = len(curve_metrics)
    if figsize is None:
        figsize = (9.5, 10.5 + 1.1 * n_metrics)

    fig = plt.figure(figsize=figsize)
    rows = fig.add_gridspec(
        4, 1, height_ratios=[1.1, 1.1, 1.3, 0.62 * n_metrics + 0.45], hspace=0.42
    )

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    model_colors = [colors[i % len(colors)] for i in range(k)]

    # Rows 1-2: overlaid mean summaries with bands, shared x.
    ax_land = fig.add_subplot(rows[0])
    ax_sil = fig.add_subplot(rows[1], sharex=ax_land)
    for ax, summary, title in (
        (ax_land, "landscape", "Mean Landscape"),
        (ax_sil, "silhouette", "Mean Silhouette"),
    ):
        for i, rep in enumerate(reps):
            curve = rep.summary(summary)
            ax.plot(curve[:, 0], curve[:, 1], color=model_colors[i], lw=1.7, label=names[i])
            band = getattr(rep, f"{summary}_band", None)
            if band is not None and len(band["lower_bound"]) == len(curve):
                ax.fill_between(
                    curve[:, 0], band["lower_bound"], band["upper_bound"],
                    color=model_colors[i], alpha=0.22, lw=0,
                )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(f"{summary.capitalize()} Value", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)
    ax_land.tick_params(labelbottom=False)
    ax_sil.set_xlabel("t", fontsize=9)
    ax_land.legend(
        title="Model", loc="center left", bbox_to_anchor=(1.01, 0.0),
        fontsize=9, title_fontsize=9, frameon=False,
    )

    # Row 3: energy statistic vs baseline, one line per summary type.
    ax_e = fig.add_subplot(rows[2])
    series_colors = {"landscape": "steelblue", "silhouette": "orange"}
    x = np.arange(k)
    for summary in summaries:
        values = [s for s, _ in energy[summary]]
        ax_e.plot(x, values, marker="o", color=series_colors[summary], label=f"{summary}s")
    tick_labels = []
    for i in range(k):
        star_pair = " ".join(
            _stars(p) if i != baseline else "ns"
            for _, p in (energy["landscape"][i], energy["silhouette"][i])
        )
        tick_labels.append(f"{names[i]}\n{star_pair}")
    ax_e.set_xticks(x, tick_labels, fontsize=9)
    ax_e.set_ylabel("Energy Statistics", fontsize=9)
    ax_e.set_xlabel(
        "Model / Star Significance Level (landscapes, silhouettes)", fontsize=9
    )
    ax_e.tick_params(labelsize=8)
    ax_e.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    ax_e.margins(x=0.08)

    # Row 4: one 2-row heatmap strip per metric (rows: silhouettes, landscapes).
    strips = rows[3].subgridspec(n_metrics, 1, hspace=0.9)
    cmap = plt.cm.YlOrBr
    last_im = None
    for m, metric in enumerate(curve_metrics):
        ax = fig.add_subplot(strips[m])
        grid_scores = np.array(
            [
                [s for s, _ in metric_scores[metric]["silhouette"]],
                [s for s, _ in metric_scores[metric]["landscape"]],
            ]
        )
        # Normalize per strip so one low->high colorbar serves all strips.
        span = grid_scores.max() - grid_scores.min()
        normed = (grid_scores - grid_scores.min()) / (span if span > 0 else 1.0)
        last_im = ax.imshow(normed, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        for row, summary in enumerate(("silhouette", "landscape")):
            for i in range(k):
                _, p = metric_scores[metric][summary][i]
                label = "ns" if i == baseline else _stars(p)
                ax.text(
                    i, row, label, ha="center", va="center", fontsize=9,
                    fontweight="bold",
                    color="black" if normed[row, i] < 0.6 else "white",
                )
        ax.set_yticks([0, 1], ["silhouettes", "landscapes"], fontsize=8)
        title = metric.replace("js", "JS").replace("wasserstein", "Wasserstein")
        ax.set_title(f"{title} Distance", fontsize=10, loc="left")
        if m == n_metrics - 1:
            ax.set_xticks(range(k), names, fontsize=9, rotation=20, ha="right")
            ax.set_xlabel("Model", fontsize=9)
        else:
            ax.set_xticks(range(k), [""] * k)

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=fig.axes[-n_metrics:], fraction=0.05, pad=0.03, ticks=[0, 1]
        )
        cbar.ax.set_yticklabels(["low", "high"], fontsize=8)

    return fig
