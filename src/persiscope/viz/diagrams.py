"""Persistence-diagram plots."""

from __future__ import annotations

import numpy as np

from ._backends import require_matplotlib


def plot_persistence_diagram(representation, *, ax=None, title: str | None = None):
    """Scatter the (birth, death) points of a representation's diagram.

    Infinite (essential) deaths are drawn at the top of the axis. Returns the
    matplotlib ``Figure``.
    """
    plt = require_matplotlib()
    diagram = np.asarray(representation.persistence_diagram, dtype=float)

    fig = ax.figure if ax is not None else plt.figure(figsize=(5, 5))
    if ax is None:
        ax = fig.add_subplot(111)

    finite = diagram[np.isfinite(diagram).all(axis=1)] if len(diagram) else diagram
    infinite = diagram[~np.isfinite(diagram).all(axis=1)] if len(diagram) else diagram

    lo = float(finite.min()) if len(finite) else 0.0
    hi = float(finite.max()) if len(finite) else 1.0
    pad = 0.05 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad

    ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, zorder=1)
    if len(finite):
        ax.scatter(finite[:, 0], finite[:, 1], s=18, alpha=0.8, zorder=2, label="finite")
    if len(infinite):
        ax.scatter(
            infinite[:, 0], np.full(len(infinite), hi), s=28, marker="^",
            color="crimson", zorder=3, label="essential",
        )
        ax.legend(loc="lower right", fontsize=8)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    label = getattr(representation, "label", None)
    ax.set_title(title or f"Persistence diagram (H{representation.homology_dim}"
                 + (f", {label}" if label else "") + ")")
    fig.tight_layout()
    return fig
