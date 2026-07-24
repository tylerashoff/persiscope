"""Swappable persistence-diagram transforms.

A *diagram transform* maps a persistence diagram (an ``(n, 2)`` array of
``(birth, death)`` points) to a new set of planar coordinates before summary
functions (landscapes, silhouettes) are built on top of it. Rotating and
scaling the diagram is what turns the "staircase" of an H0 diagram into the
tent functions the landscape machinery expects.

This is a deliberately small, self-contained surface: transforms are plain
callables with a single ``__call__(diagram, homology_dim)`` method, so new
coordinate schemes can be dropped in and compared without touching the rest of
the pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DiagramTransform(Protocol):
    """Anything that maps an ``(n, 2)`` diagram to an ``(n, 2)`` diagram."""

    def __call__(self, diagram: np.ndarray, homology_dim: int = 0) -> np.ndarray: ...


class Identity:
    """Pass the diagram through unchanged."""

    def __call__(self, diagram: np.ndarray, homology_dim: int = 0) -> np.ndarray:
        return np.asarray(diagram, dtype=float)

    def __repr__(self) -> str:
        return "Identity()"


class RotateScale:
    """Rotate by ``theta`` then uniformly scale by ``alpha``.

    ``transformed = alpha * (R(theta) @ diagram.T).T``

    The defaults (``theta = -3*pi/8``, ``alpha = sqrt(2)/2``) reproduce the
    transform used by the reference TDA pipeline: they send the diagonal to the
    horizontal axis and rescale so a maximum pairwise distance of 1 maps to a
    maximum landscape support of ``sqrt(2)/2``.
    """

    def __init__(self, theta: float = -3 * np.pi / 8, alpha: float = np.sqrt(2) / 2):
        self.theta = float(theta)
        self.alpha = float(alpha)

    def __call__(self, diagram: np.ndarray, homology_dim: int = 0) -> np.ndarray:
        diagram = np.asarray(diagram, dtype=float)
        if diagram.size == 0:
            return diagram.reshape(0, 2)
        rotation = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )
        return self.alpha * np.dot(rotation, diagram.T).T

    def __repr__(self) -> str:
        return f"RotateScale(theta={self.theta:.6g}, alpha={self.alpha:.6g})"


class H0Rotate:
    """Rotate the vertical death axis to ``angle`` from horizontal.

    ``transformed = diagram @ R(pi/2 - angle)``

    An alternative H0 coordinate scheme (the "birth vs. rotated death" view).
    ``angle`` is measured counter-clockwise from the positive x-axis; the
    default ``3*pi/8`` tilts the death axis just off vertical.
    """

    def __init__(self, angle: float = 3 * np.pi / 8):
        self.angle = float(angle)

    def __call__(self, diagram: np.ndarray, homology_dim: int = 0) -> np.ndarray:
        diagram = np.asarray(diagram, dtype=float)
        if diagram.size == 0:
            return diagram.reshape(0, 2)
        a = np.pi / 2 - self.angle
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return np.dot(diagram, rotation)

    def __repr__(self) -> str:
        return f"H0Rotate(angle={self.angle:.6g})"
