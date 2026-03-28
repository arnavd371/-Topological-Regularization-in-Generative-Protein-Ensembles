"""topo_loss.py

Differentiable topological regularization loss based on H1 persistent
homology of protein backbone Cα coordinates.

Surrogate-Gradient Approach
----------------------------
``gudhi`` computes the persistence diagram from a distance matrix and
is *not* differentiable with respect to the input point coordinates.
To obtain a differentiable loss we use a **surrogate gradient** strategy:

1. Compute the persistence diagram via gudhi on ``coords_np``
   (a plain NumPy array detached from the computation graph).
2. Identify the *critical birth edges* — the 1-simplex (i, j) whose
   filtration value equals the birth time of each significant H1 bar.
3. Re-compute the length of each critical edge using
   ``coords_tensor[i] - coords_tensor[j]`` (the original PyTorch
   tensor with ``requires_grad=True``).
4. The loss is the (weighted) mean of these edge lengths.

Because step 3 uses ``coords_tensor``, gradients flow back into the
coordinates through the edge-length computation.  Minimising this loss
pushes the critical birth edges to be shorter, which in turn prevents
the formation of spurious loops in the backbone conformation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from topotraj.topology.backbone_complex import coords_to_simplex_tree
from topotraj.topology.persistence import compute_h1_persistence


def _zero_loss(
    coords_tensor: torch.Tensor,
) -> torch.Tensor:
    """Return a zero scalar loss with matching dtype and device.

    Parameters
    ----------
    coords_tensor:
        Reference tensor used to determine dtype and device.

    Returns
    -------
    torch.Tensor
        Scalar zero tensor (no gradient).
    """
    return torch.tensor(0.0, dtype=coords_tensor.dtype, device=coords_tensor.device)


def find_critical_edges(
    coords: np.ndarray,
    simplex_tree,
    bars: List[Tuple[float, float]],
) -> List[Tuple[int, int]]:
    """Find the birth-edge (critical 1-simplex) for each H1 bar.

    The birth of an H1 bar in the Rips filtration is determined by the
    edge whose addition to the filtration first closes a loop.  This
    function approximates that edge by finding the 1-simplex in the
    complex whose filtration value (edge length) is closest to the
    birth value of each bar.

    Parameters
    ----------
    coords:
        NumPy array of shape ``(N, 3)`` – Cα coordinates.
    simplex_tree:
        The ``gudhi.SimplexTree`` used to compute the persistence.
    bars:
        List of ``(birth, death)`` tuples (finite H1 bars).

    Returns
    -------
    List[Tuple[int, int]]
        One ``(i, j)`` vertex-index pair per bar, identifying the
        1-simplex (edge) whose filtration value is closest to the
        bar's birth time.  Duplicate edges are allowed.
    """
    # Collect all edges (1-simplices) and their filtration values
    edges: List[Tuple[Tuple[int, int], float]] = []
    for simplex, filtration_value in simplex_tree.get_filtration():
        if len(simplex) == 2:
            i, j = simplex
            edges.append(((i, j), filtration_value))

    if not edges:
        # Fallback: return (0, 1) for every bar when no edges are found
        return [(0, 1)] * len(bars)

    critical: List[Tuple[int, int]] = []
    for birth, _death in bars:
        # Find the edge whose filtration value is closest to the birth
        best_edge, best_diff = edges[0][0], abs(edges[0][1] - birth)
        for (i, j), fval in edges[1:]:
            diff = abs(fval - birth)
            if diff < best_diff:
                best_diff = diff
                best_edge = (i, j)
        critical.append(best_edge)

    return critical


def topological_loss(
    coords_tensor: torch.Tensor,
    max_edge_length: float = 8.0,
    persistence_threshold: float = 1.0,
    loss_weight: float = 0.1,
) -> torch.Tensor:
    """Compute a differentiable topological regularization loss.

    The loss penalizes significant H1 persistent-homology features
    (spurious loops) in the Cα backbone by targeting the critical
    birth edges that give rise to each loop.

    Surrogate-gradient note
    -----------------------
    The persistence diagram is computed from ``coords_np`` (a NumPy
    view, no grad).  Edge lengths for the loss are then recomputed
    from ``coords_tensor`` (the differentiable PyTorch tensor), so
    that gradients propagate back to the coordinates.

    Parameters
    ----------
    coords_tensor:
        Tensor of shape ``(N, 3)`` with ``requires_grad=True``.
    max_edge_length:
        Maximum edge length (Å) for the Rips complex.
    persistence_threshold:
        Minimum bar lifetime ``(death - birth)`` to be considered a
        significant loop worth penalising.
    loss_weight:
        Scalar multiplier applied to the aggregated edge-length sum.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor, differentiable with respect to
        ``coords_tensor``.  Returns ``torch.tensor(0.0)`` (no grad)
        when no significant H1 bars are found.
    """
    # Step 1: detach coordinates for gudhi (no differentiability here)
    coords_np = coords_tensor.detach().cpu().numpy()

    # Step 2: build simplex tree and compute persistence
    simplex_tree = coords_to_simplex_tree(
        coords_np,
        max_edge_length=max_edge_length,
        max_dimension=2,
    )
    bars = compute_h1_persistence(simplex_tree)

    # Step 3: filter to significant bars only
    significant_bars = [
        (b, d) for b, d in bars if (d - b) > persistence_threshold
    ]

    # Edge case: no significant loops — return zero loss
    if not significant_bars:
        return _zero_loss(coords_tensor)

    # Step 4: identify the critical birth edges for significant bars
    # (uses coords_np / simplex_tree — no grad)
    critical_edges = find_critical_edges(coords_np, simplex_tree, significant_bars)

    # Step 5: compute edge lengths from coords_tensor so gradients flow.
    # This is the surrogate: minimising these lengths discourages the
    # backbone from forming the detected loops.
    edge_lengths = []
    for i, j in critical_edges:
        # edge_length is differentiable w.r.t. coords_tensor
        edge_length = torch.norm(coords_tensor[i] - coords_tensor[j])
        edge_lengths.append(edge_length)

    num_significant = max(1, len(significant_bars))
    loss = loss_weight * torch.stack(edge_lengths).sum() / num_significant
    return loss


def trajectory_topological_loss(
    trajectory: torch.Tensor,
    max_edge_length: float = 8.0,
    persistence_threshold: float = 1.0,
    loss_weight: float = 0.1,
) -> torch.Tensor:
    """Compute the mean topological loss over intermediate trajectory frames.

    Applies :func:`topological_loss` to every frame except the first
    and last (endpoints are fixed; only intermediate conformations are
    penalised).

    Parameters
    ----------
    trajectory:
        Tensor of shape ``(T, N, 3)`` representing a sequence of
        conformer frames.  Requires grad on the intermediate frames.
    max_edge_length:
        Maximum edge length (Å) for the Rips complex.
    persistence_threshold:
        Minimum bar lifetime to penalise.
    loss_weight:
        Scalar multiplier for the loss.

    Returns
    -------
    torch.Tensor
        Scalar mean topological loss across intermediate frames.
        Returns ``torch.tensor(0.0)`` if the trajectory has fewer than
        3 frames (no intermediate frames).
    """
    T = trajectory.shape[0]

    # Need at least one intermediate frame
    if T < 3:
        return _zero_loss(trajectory)

    frame_losses: List[torch.Tensor] = []
    for t in range(1, T - 1):  # skip first (t=0) and last (t=T-1)
        frame_loss = topological_loss(
            trajectory[t],
            max_edge_length=max_edge_length,
            persistence_threshold=persistence_threshold,
            loss_weight=loss_weight,
        )
        frame_losses.append(frame_loss)

    return torch.stack(frame_losses).mean()
