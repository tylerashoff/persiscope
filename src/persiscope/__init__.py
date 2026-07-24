"""persiscope: topological representations and pairwise comparison of embeddings.

*persistence + periscope.* Turn embedded data into persistence-based summaries
(landscapes and silhouettes) and score how topologically similar two embedding
sets are.

Quickstart
----------
>>> import persiscope as ps
>>> result = ps.compare([emb_a, emb_b, emb_c], method="energy")   # doctest: +SKIP
>>> result.matrix                                                 # doctest: +SKIP
"""

from __future__ import annotations

from . import diagram_transforms as transforms
from . import viz
from ._serialize import load, save
from .compare import compare
from .diagram_transforms import DiagramTransform, H0Rotate, Identity, RotateScale
from .graph import build_graph
from .persistence import Persistence, compute_persistence
from .representation import Representation, TopologicalTransformer, transform
from .scoring import (
    CURVE_METRICS,
    ComparisonResult,
    ScoreMatrix,
    Scorer,
    ScoreResult,
    energy_statistic,
    get_curve_metric,
    permutation_pvalue,
    score,
    score_matrix,
)
from .summaries import Summaries, compute_summaries

__version__ = "0.1.0b4"

__all__ = [
    "CURVE_METRICS",
    "ComparisonResult",
    "DiagramTransform",
    "H0Rotate",
    "Identity",
    "Persistence",
    "Representation",
    "RotateScale",
    "ScoreMatrix",
    "ScoreResult",
    # scoring
    "Scorer",
    "Summaries",
    # representation
    "TopologicalTransformer",
    "__version__",
    "build_graph",
    # main entry point
    "compare",
    "compute_persistence",
    "compute_summaries",
    "energy_statistic",
    "get_curve_metric",
    "load",
    "permutation_pvalue",
    "save",
    "score",
    "score_matrix",
    "transform",
    # diagram transforms
    "transforms",
    # viz + io
    "viz",
]
