"""train_topo.py

Training routine for the topologically-regularized flow model
(:class:`~topotraj.models.flow_topo.TopoFlowModel`).

Extends the baseline training loop with a topological regularization
term: after computing the standard flow-matching loss, the model
integrates a short trajectory from the current source frame and the
topological loss is computed on that trajectory.  The total loss is

    total_loss = flow_loss + topo_loss

Key design note
---------------
The topological loss is computed on ``trajectory`` which is produced
by :meth:`model.integrate` **inside the training loop with gradients
enabled**.  The surrogate-gradient mechanism in
:func:`~topotraj.topology.topo_loss.topological_loss` re-computes
edge lengths from the trajectory tensors so that gradients propagate
back into the model parameters.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from topotraj.models.flow_topo import TopoFlowModel
from topotraj.data.interpolate import linear_interpolation, get_frame_index
from topotraj.topology.topo_loss import trajectory_topological_loss


def _integrate_with_grad(
    model: TopoFlowModel,
    source: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Euler-integrate the model's velocity field *with gradients enabled*.

    This is a gradient-aware version of :meth:`integrate` that keeps
    the computational graph intact so that the topological loss can
    propagate gradients back through the trajectory into the model.

    Parameters
    ----------
    model:
        The flow model being trained.
    source:
        Source conformation of shape ``(N, 3)``.
    num_steps:
        Number of Euler steps.

    Returns
    -------
    torch.Tensor
        Trajectory of shape ``(num_steps + 1, N, 3)`` with the
        computational graph attached to intermediate frames.
    """
    device = source.device
    dt = 1.0 / num_steps

    frames: list[torch.Tensor] = [source]
    coords = source

    for step in range(num_steps):
        t_val = torch.tensor(step * dt, dtype=source.dtype, device=device)
        velocity = model(coords, t_val)
        coords = coords + dt * velocity
        frames.append(coords)

    return torch.stack(frames, dim=0)  # (num_steps+1, N, 3)


def train_topo(config: dict) -> TopoFlowModel:
    """Train the topologically-regularized flow model.

    Same as :func:`~topotraj.training.train_baseline.train_baseline`
    but augments the flow-matching loss with a trajectory topological
    loss that penalises spurious H1 loops in the generated path.

    Parameters
    ----------
    config:
        Configuration dictionary.  Expected keys (same as baseline
        plus topo-specific keys):

        * ``"train_pairs"`` — list of ``(coords_A, coords_B)`` pairs.
        * ``"training_steps"`` — total gradient steps.
        * ``"flow_lr"`` — Adam learning rate.
        * ``"flow_hidden_dim"`` — hidden layer width (default 256).
        * ``"num_interpolation_steps"`` — interpolation frames.
        * ``"max_edge_length"`` — Rips complex edge threshold (Å).
        * ``"persistence_threshold"`` — minimum bar lifetime to penalise.
        * ``"topo_loss_weight"`` — scalar weight for the topo loss.
        * ``"num_integration_steps"`` — steps for trajectory integration
          during training (default 10, kept small for speed).

    Returns
    -------
    TopoFlowModel
        The trained model in evaluation mode.
    """
    train_pairs: List[Tuple[np.ndarray, np.ndarray]] = config["train_pairs"]
    training_steps: int = int(config.get("training_steps", 10000))
    lr: float = float(config.get("flow_lr", 1e-4))
    hidden_dim: int = int(config.get("flow_hidden_dim", 256))
    num_interp: int = int(config.get("num_interpolation_steps", 20))
    max_edge_length: float = float(config.get("max_edge_length", 8.0))
    persistence_threshold: float = float(config.get("persistence_threshold", 1.0))
    topo_weight: float = float(config.get("topo_loss_weight", 0.1))
    num_traj_steps: int = int(config.get("num_integration_steps", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_residues = train_pairs[0][0].shape[0]

    model = TopoFlowModel(n_residues=n_residues, hidden_dim=hidden_dim)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for step in range(1, training_steps + 1):
        # Sample a random conformer pair
        coords_A, coords_B = random.choice(train_pairs)

        t_float = random.uniform(0.0, 1.0)
        t = torch.tensor(t_float, dtype=torch.float32, device=device)

        trajectory_np = linear_interpolation(coords_A, coords_B, num_steps=num_interp)
        frame_idx = get_frame_index(t_float, num_interp)

        coords_at_t = torch.tensor(
            trajectory_np[frame_idx], dtype=torch.float32, device=device
        )
        target_velocity = torch.tensor(
            coords_B - coords_A, dtype=torch.float32, device=device
        )

        # --- Flow-matching loss ---
        pred_velocity = model(coords_at_t, t)
        flow_loss = criterion(pred_velocity, target_velocity)

        # --- Topological loss ---
        # Integrate a short trajectory WITH GRADIENTS so the topo loss
        # can propagate back into the model parameters.
        source_tensor = torch.tensor(
            coords_A, dtype=torch.float32, device=device
        ).requires_grad_(False)

        integrated_traj = _integrate_with_grad(model, source_tensor, num_traj_steps)
        # integrated_traj: (num_traj_steps+1, N, 3) — differentiable

        topo_loss = trajectory_topological_loss(
            integrated_traj,
            max_edge_length=max_edge_length,
            persistence_threshold=persistence_threshold,
            loss_weight=topo_weight,
        )

        total_loss = flow_loss + topo_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"[TopoFlow] Step {step:5d}/{training_steps}  "
                f"flow_loss={flow_loss.item():.6f}  "
                f"topo_loss={topo_loss.item():.6f}  "
                f"total={total_loss.item():.6f}"
            )

    model.eval()
    return model
