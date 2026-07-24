"""Lazy backend imports with actionable error messages."""

from __future__ import annotations


def require_matplotlib():
    """Return ``matplotlib.pyplot`` or raise a helpful ImportError."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "This plot needs matplotlib. Install the viz extra with "
            "`pip install persiscope[viz]`."
        ) from exc
    return plt


def require_plotly():
    """Return ``plotly.graph_objects`` or raise a helpful ImportError."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError(
            "This plot needs plotly. Install the viz extra with "
            "`pip install persiscope[viz]`."
        ) from exc
    return go
