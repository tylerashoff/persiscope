"""Optional visualization helpers (install with ``pip install persiscope[viz]``).

Every function returns a figure object (matplotlib ``Figure`` or plotly
``Figure``); the caller decides whether to show or save it. Backends are
imported lazily so the core package has no plotting dependency.
"""

from .diagrams import plot_persistence_diagram
from .report import plot_baseline_report, plot_comparison_report
from .scores import plot_score_heatmap
from .summaries import plot_landscape, plot_silhouette, plot_summary

__all__ = [
    "plot_baseline_report",
    "plot_comparison_report",
    "plot_landscape",
    "plot_persistence_diagram",
    "plot_score_heatmap",
    "plot_silhouette",
    "plot_summary",
]
