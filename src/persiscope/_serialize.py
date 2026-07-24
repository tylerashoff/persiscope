"""Opt-in save/load for representations (NPZ). Not used by the core pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .representation import Representation

PathLike = str | Path


def save(representation: Representation, path: PathLike) -> None:
    """Write a :class:`Representation` to a compressed ``.npz`` file."""
    r = representation
    np.savez_compressed(
        path,
        homology_dim=r.homology_dim,
        persistence_diagram=r.persistence_diagram,
        finite_diagram=r.finite_diagram,
        transformed_diagram=r.transformed_diagram,
        full_landscape=r.full_landscape,
        full_silhouette=r.full_silhouette,
        mean_landscape=r.mean_landscape,
        mean_silhouette=r.mean_silhouette,
        bootstrapped_landscapes=r.bootstrapped_landscapes,
        bootstrapped_silhouettes=r.bootstrapped_silhouettes,
        skeleton1=r.skeleton1,
        landscape_band=np.array(r.landscape_band, dtype=object),
        silhouette_band=np.array(r.silhouette_band, dtype=object),
        sampled_node_indices=np.array(r.sampled_node_indices, dtype=object),
        normalization_factor=np.array(r.normalization_factor, dtype=object),
        params=np.array(r.params, dtype=object),
        label=np.array(r.label, dtype=object),
    )


def load(path: PathLike) -> Representation:
    """Load a :class:`Representation` previously written with :func:`save`."""
    data = np.load(path, allow_pickle=True)
    return Representation(
        homology_dim=int(data["homology_dim"]),
        persistence_diagram=data["persistence_diagram"],
        finite_diagram=data["finite_diagram"],
        transformed_diagram=data["transformed_diagram"],
        full_landscape=data["full_landscape"],
        full_silhouette=data["full_silhouette"],
        mean_landscape=data["mean_landscape"],
        mean_silhouette=data["mean_silhouette"],
        bootstrapped_landscapes=data["bootstrapped_landscapes"],
        bootstrapped_silhouettes=data["bootstrapped_silhouettes"],
        landscape_band=data["landscape_band"].item(),
        silhouette_band=data["silhouette_band"].item(),
        skeleton1=data["skeleton1"],
        sampled_node_indices=list(data["sampled_node_indices"]),
        normalization_factor=data["normalization_factor"].item(),
        params=data["params"].item(),
        label=data["label"].item(),
    )
