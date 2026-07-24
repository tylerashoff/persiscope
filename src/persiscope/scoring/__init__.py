"""Pairwise scoring of topological representations."""

from .distance_metrics import CURVE_METRICS, get_curve_metric
from .landscape_scores import energy_statistic
from .permutation import permutation_pvalue
from .results import ComparisonResult, ScoreMatrix, ScoreResult
from .scorer import Scorer, score, score_matrix

__all__ = [
    "CURVE_METRICS",
    "ComparisonResult",
    "ScoreMatrix",
    "ScoreResult",
    "Scorer",
    "energy_statistic",
    "get_curve_metric",
    "permutation_pvalue",
    "score",
    "score_matrix",
]
