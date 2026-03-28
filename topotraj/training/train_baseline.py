"""train_baseline.py

Training routine for the baseline flow-matching model
(:class:`~topotraj.models.flow_baseline.ConformerFlowModel`).

The training objective is a standard flow-matching loss: at a random
interpolation time ``t``, the model predicts the velocity field that
should move the source conformation toward the target, and the loss is
the mean-squared error between the predicted velocity and the
analytic (linear interpolation) target velocity
``v* = coords_target - coords_source``.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from topotraj.models.flow_baseline import ConformerFlowModel
from topotraj.data.interpolate import linear_interpolation, get_frame_index


def train_baseline(config: dict) -> ConformerFlowModel:
    """Train the baseline flow-matching model without topological loss.

    For each training step a random conformer pair is sampled from the
    dataset, a random interpolation time ``t ~ U(0, 1)`` is drawn, and
    the model is trained to predict the target velocity
    ``(coords_target - coords_source)``.

    Parameters
    ----------
    config:
        Configuration dictionary.  Expected keys:

        * ``"train_pairs"`` — list of ``(coords_A, coords_B)`` NumPy
          array pairs (injected by the caller, not read from disk here).
        * ``"training_steps"`` — total number of gradient steps.
        * ``"flow_lr"`` — Adam learning rate.
        * ``"flow_hidden_dim"`` — hidden layer width (default 256).
        * ``"num_interpolation_steps"`` — number of interpolation
          frames used to build target trajectories (default 20).

    Returns
    -------
    ConformerFlowModel
        The trained model in evaluation mode.
    """
    train_pairs: List[Tuple[np.ndarray, np.ndarray]] = config["train_pairs"]
    training_steps: int = int(config.get("training_steps", 10000))
    lr: float = float(config.get("flow_lr", 1e-4))
    hidden_dim: int = int(config.get("flow_hidden_dim", 256))
    num_interp: int = int(config.get("num_interpolation_steps", 20))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer number of residues from the first pair
    n_residues = train_pairs[0][0].shape[0]

    model = ConformerFlowModel(n_residues=n_residues, hidden_dim=hidden_dim)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for step in range(1, training_steps + 1):
        # Sample a random conformer pair
        coords_A, coords_B = random.choice(train_pairs)

        # Random interpolation time in [0, 1]
        t_float = random.uniform(0.0, 1.0)
        t = torch.tensor(t_float, dtype=torch.float32, device=device)

        # Build the linear trajectory and sample a frame at time t
        trajectory = linear_interpolation(coords_A, coords_B, num_steps=num_interp)
        frame_idx = get_frame_index(t_float, num_interp)

        coords_at_t = torch.tensor(
            trajectory[frame_idx], dtype=torch.float32, device=device
        )

        # Target velocity: direction from source to target (constant for linear flow)
        target_velocity = torch.tensor(
            coords_B - coords_A, dtype=torch.float32, device=device
        )

        # Forward pass
        pred_velocity = model(coords_at_t, t)

        loss = criterion(pred_velocity, target_velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[Baseline] Step {step:5d}/{training_steps}  loss={loss.item():.6f}")

    model.eval()
    return model
