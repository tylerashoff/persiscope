"""Landscape and silhouette curve plots with confidence-band shading."""

from __future__ import annotations

import numpy as np

from ._backends import require_matplotlib


def plot_summary(
    representation,
    kind: str = "landscape",
    *,
    mean: bool = True,
    band: bool = True,
    ax=None,
    title: str | None = None,
):
    """Plot a representation's ``landscape`` or ``silhouette`` curve.

    With ``band=True`` and ``mean=True`` the matching confidence band is shaded.
    Returns the matplotlib ``Figure``.
    """
    plt = require_matplotlib()
    if kind not in ("landscape", "silhouette"):
        raise ValueError(f"kind must be 'landscape' or 'silhouette', got {kind!r}.")

    curve = representation.summary(kind, mean=mean)
    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 3.5))
    if ax is None:
        ax = fig.add_subplot(111)

    t, y = curve[:, 0], curve[:, 1]
    ax.plot(t, y, lw=1.6, label=f"mean {kind}" if mean else f"{kind}")

    if band and mean:
        band_dict = getattr(representation, f"{kind}_band", None)
        if band_dict is not None:
            lower = np.asarray(band_dict["lower_bound"])
            upper = np.asarray(band_dict["upper_bound"])
            if len(lower) == len(t):
                ax.fill_between(t, lower, upper, alpha=0.2, label="confidence band")

    ax.set_xlabel("t")
    ax.set_ylabel(kind)
    label = getattr(representation, "label", None)
    ax.set_title(title or f"Persistence {kind} (H{representation.homology_dim}"
                 + (f", {label}" if label else "") + ")")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_landscape(representation, **kwargs):
    """Plot the persistence landscape (see :func:`plot_summary`)."""
    return plot_summary(representation, kind="landscape", **kwargs)


def plot_silhouette(representation, **kwargs):
    """Plot the persistence silhouette (see :func:`plot_summary`)."""
    return plot_summary(representation, kind="silhouette", **kwargs)
