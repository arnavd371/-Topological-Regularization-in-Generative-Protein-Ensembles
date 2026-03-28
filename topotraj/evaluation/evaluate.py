"""evaluate.py

Full evaluation runner for generated conformational trajectories.

Computes a battery of metrics (steric clash rate, endpoint RMSD, and
mean H1 persistence) for a list of source–target pairs and returns
per-metric means and standard deviations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from topotraj.evaluation.steric_clash import trajectory_clash_rate
from topotraj.evaluation.rmsd_metrics import compute_rmsd
from topotraj.topology.backbone_complex import coords_to_simplex_tree
from topotraj.topology.persistence import (
    compute_h1_persistence,
    total_h1_persistence,
)


def _mean_h1_trajectory(
    trajectory: np.ndarray,
    max_edge_length: float = 8.0,
) -> float:
    """Compute mean total H1 persistence across all trajectory frames.

    Parameters
    ----------
    trajectory:
        Array of shape ``(T, N, 3)``.
    max_edge_length:
        Rips complex edge threshold in Å.

    Returns
    -------
    float
        Mean total H1 persistence (sum of bar lifetimes) averaged over
        all frames.
    """
    persistences: List[float] = []
    for t in range(trajectory.shape[0]):
        st = coords_to_simplex_tree(trajectory[t], max_edge_length=max_edge_length)
        bars = compute_h1_persistence(st)
        persistences.append(total_h1_persistence(bars))
    return float(np.mean(persistences))


def evaluate_model(
    model: Any,
    test_pairs: List[Tuple[np.ndarray, np.ndarray]],
    num_steps: int = 50,
    max_edge_length: float = 8.0,
    clash_threshold: float = 2.0,
) -> Dict[str, float]:
    """Evaluate a flow model on a set of test conformer pairs.

    For each ``(source, target)`` pair the model generates a trajectory
    via :meth:`integrate`, then the following metrics are computed:

    * **Clash rate** — mean number of steric clashes per frame
      (``trajectory_clash_rate``).
    * **Endpoint RMSD** — RMSD between the final generated frame and
      the ground-truth target.
    * **Mean H1 persistence** — average total H1 persistence across all
      trajectory frames (a proxy for spurious-loop content).

    Parameters
    ----------
    model:
        A trained flow model with an ``integrate(source, num_steps)``
        method that returns a ``torch.Tensor`` of shape
        ``(T, N, 3)``.
    test_pairs:
        List of ``(coords_source, coords_target)`` NumPy array pairs.
    num_steps:
        Number of Euler integration steps for trajectory generation.
    max_edge_length:
        Rips complex edge threshold (Å) for H1 computation.
    clash_threshold:
        Minimum Cα–Cα distance (Å) to define a steric clash.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        ``clash_rate_mean``, ``clash_rate_std``,
        ``endpoint_rmsd_mean``, ``endpoint_rmsd_std``,
        ``mean_h1_persistence_mean``, ``mean_h1_persistence_std``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clash_rates: List[float] = []
    endpoint_rmsds: List[float] = []
    h1_persistences: List[float] = []

    model.eval()
    with torch.no_grad():
        for coords_source, coords_target in test_pairs:
            source_tensor = torch.tensor(
                coords_source, dtype=torch.float32, device=device
            )

            # Generate trajectory
            traj_tensor = model.integrate(source_tensor, num_steps=num_steps)
            traj_np = traj_tensor.cpu().numpy()  # (T, N, 3)

            # Steric clash rate
            clash_rates.append(
                trajectory_clash_rate(traj_np, clash_threshold=clash_threshold)
            )

            # Endpoint RMSD to target
            final_frame = traj_np[-1]  # (N, 3)
            n = min(final_frame.shape[0], coords_target.shape[0])
            endpoint_rmsds.append(compute_rmsd(final_frame[:n], coords_target[:n]))

            # Mean H1 persistence across trajectory
            h1_persistences.append(
                _mean_h1_trajectory(traj_np, max_edge_length=max_edge_length)
            )

    return {
        "clash_rate_mean": float(np.mean(clash_rates)),
        "clash_rate_std": float(np.std(clash_rates)),
        "endpoint_rmsd_mean": float(np.mean(endpoint_rmsds)),
        "endpoint_rmsd_std": float(np.std(endpoint_rmsds)),
        "mean_h1_persistence_mean": float(np.mean(h1_persistences)),
        "mean_h1_persistence_std": float(np.std(h1_persistences)),
    }
