"""persistence.py

Functions for computing H1 persistent homology from a gudhi
``SimplexTree`` and converting the resulting persistence bars into
PyTorch tensors and scalar summaries.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import gudhi


def compute_h1_persistence(
    simplex_tree: gudhi.SimplexTree,
) -> List[Tuple[float, float]]:
    """Compute persistent homology and return finite H1 bars.

    Calls ``simplex_tree.compute_persistence()`` and collects all
    dimension-1 (H1) features that have a finite death time.  Bars
    with ``death == float('inf')`` represent persistent generators
    that were never killed; they are excluded because they correspond
    to topological features of the Rips filtration itself rather than
    loops in the backbone path.

    Parameters
    ----------
    simplex_tree:
        A ``gudhi.SimplexTree`` that has been expanded to at least
        dimension 2 (see :func:`backbone_complex.coords_to_simplex_tree`).

    Returns
    -------
    List[Tuple[float, float]]
        List of ``(birth, death)`` pairs for finite H1 features.
        An empty list means no loops were detected.
    """
    simplex_tree.compute_persistence()
    all_pairs = simplex_tree.persistence()

    bars: List[Tuple[float, float]] = []
    for dimension, (birth, death) in all_pairs:
        if dimension == 1 and death != float("inf"):
            bars.append((birth, death))

    return bars


def persistence_to_tensor(
    bars: List[Tuple[float, float]],
) -> torch.Tensor:
    """Convert a list of persistence bars to a 2-D tensor.

    Parameters
    ----------
    bars:
        List of ``(birth, death)`` tuples produced by
        :func:`compute_h1_persistence`.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N_bars, 2)`` where column 0 is birth and
        column 1 is death.  If *bars* is empty, returns an empty
        tensor of shape ``(0, 2)``.
    """
    if not bars:
        return torch.zeros((0, 2), dtype=torch.float32)
    return torch.tensor(bars, dtype=torch.float32)


def total_h1_persistence(
    bars: List[Tuple[float, float]],
) -> float:
    """Compute the total (summed) persistence of all H1 bars.

    Total persistence is defined as

        sum_i (death_i - birth_i)

    A larger value indicates more extensive looping structure in the
    backbone conformation.

    Parameters
    ----------
    bars:
        List of ``(birth, death)`` tuples.

    Returns
    -------
    float
        Sum of bar lifetimes.  Returns ``0.0`` for an empty list.
    """
    return sum(death - birth for birth, death in bars)
