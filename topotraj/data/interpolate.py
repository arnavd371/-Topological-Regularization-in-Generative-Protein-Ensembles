"""interpolate.py

Functions for generating linear interpolation trajectories between
two protein conformations.  These trajectories serve as the "ground
truth" (straight-line) paths used to train the flow-matching model.
"""

from __future__ import annotations

import numpy as np


def get_frame_index(t_float: float, num_steps: int) -> int:
    """Convert a continuous time value to a trajectory frame index.

    Parameters
    ----------
    t_float:
        Time in ``[0, 1]``.
    num_steps:
        Total number of frames in the trajectory.

    Returns
    -------
    int
        Clamped integer frame index in ``[0, num_steps - 1]``.
    """
    return max(0, min(int(t_float * (num_steps - 1)), num_steps - 1))


def linear_interpolation(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    num_steps: int = 20,
) -> np.ndarray:
    """Generate a linear interpolation trajectory between two conformations.

    For each intermediate step ``k`` the interpolated coordinates are

        coords_k = (1 - alpha) * coords_A + alpha * coords_B

    where ``alpha = k / (num_steps - 1)`` runs from 0 (= coords_A)
    to 1 (= coords_B).

    Parameters
    ----------
    coords_A:
        Source conformation, shape ``(N, 3)``.
    coords_B:
        Target conformation, shape ``(N, 3)``.
    num_steps:
        Total number of frames including the endpoints.  Must be
        at least 2.

    Returns
    -------
    np.ndarray
        Trajectory array of shape ``(num_steps, N, 3)``.  The first
        frame equals ``coords_A`` and the last frame equals
        ``coords_B``.

    Raises
    ------
    ValueError
        If ``num_steps < 2``.
    """
    if num_steps < 2:
        raise ValueError(
            f"num_steps must be at least 2, got {num_steps}."
        )

    alphas = np.linspace(0.0, 1.0, num_steps)  # (num_steps,)
    # Expand dims for broadcasting: alphas → (num_steps, 1, 1)
    alphas = alphas[:, np.newaxis, np.newaxis]

    trajectory = (1.0 - alphas) * coords_A[np.newaxis] + alphas * coords_B[np.newaxis]
    return trajectory.astype(np.float64)
