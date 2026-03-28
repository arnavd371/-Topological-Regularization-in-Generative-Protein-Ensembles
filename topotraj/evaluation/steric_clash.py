"""steric_clash.py

Functions for measuring steric clashes in protein conformations.

A steric clash is defined as a pair of Cα atoms ``(i, j)`` with
``|i - j| > 2`` (non-adjacent in sequence) whose Euclidean distance
falls below a threshold (default 2.0 Å).  Adjacent residues along the
backbone (``|i - j| <= 2``) are always close and are therefore
excluded to avoid inflating the clash count with physically normal
contacts.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def count_steric_clashes(
    coords: np.ndarray,
    clash_threshold: float = 2.0,
) -> int:
    """Count steric clashes in a single conformation.

    A clash is counted for every pair ``(i, j)`` where:
    * ``|i - j| > 2`` (backbone neighbors excluded), and
    * ``dist(coords[i], coords[j]) < clash_threshold``.

    The exclusion of ``|i - j| <= 2`` is critical: adjacent Cα atoms
    are separated by ~3.8 Å and would always be flagged as clashing
    otherwise, making the metric meaningless.

    Parameters
    ----------
    coords:
        Cα coordinates of shape ``(N, 3)``.
    clash_threshold:
        Minimum allowed Cα–Cα distance (Å).  Pairs closer than this
        are counted as clashing (default: 2.0 Å).

    Returns
    -------
    int
        Number of clashing pairs in the conformation.
    """
    N = coords.shape[0]
    if N < 4:
        # Cannot have non-adjacent clashes with fewer than 4 residues
        return 0

    dist_matrix = cdist(coords, coords, metric="euclidean")  # (N, N)

    clash_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if abs(i - j) > 2 and dist_matrix[i, j] < clash_threshold:
                clash_count += 1

    return clash_count


def trajectory_clash_rate(
    trajectory: np.ndarray,
    clash_threshold: float = 2.0,
) -> float:
    """Compute the mean steric clash count across all frames.

    This is the key physical-validity evaluation metric: a lower value
    indicates that the generated trajectory avoids non-physical
    backbone crossings.

    Parameters
    ----------
    trajectory:
        Trajectory array of shape ``(T, N, 3)``.
    clash_threshold:
        Minimum allowed Cα–Cα distance (Å) (default: 2.0 Å).

    Returns
    -------
    float
        Mean number of clashing pairs averaged across all ``T`` frames.
    """
    T = trajectory.shape[0]
    clash_counts = [
        count_steric_clashes(trajectory[t], clash_threshold=clash_threshold)
        for t in range(T)
    ]
    return float(np.mean(clash_counts))
