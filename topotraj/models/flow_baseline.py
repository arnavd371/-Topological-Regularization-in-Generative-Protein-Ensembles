"""flow_baseline.py

Simple flow-matching model for protein conformational interpolation.
The model learns a velocity field ``v(coords, t)`` such that
integrating from ``t=0`` to ``t=1`` transports the source conformation
toward the target conformation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConformerFlowModel(nn.Module):
    """Flow-matching velocity model for Cα conformational interpolation.

    Architecture
    ------------
    Input:  concatenation of flattened Cα coordinates ``(N*3,)`` and a
            scalar timestep ``t`` → vector of size ``3N + 1``.
    Hidden: three fully-connected layers of dimension ``hidden_dim``
            with SiLU (Swish) activations.
    Output: predicted velocity field of shape ``(N, 3)``, reshaped
            from a linear projection of size ``3N``.

    Parameters
    ----------
    n_residues:
        Number of Cα atoms (residues) in the protein.
    hidden_dim:
        Width of each hidden layer (default: 256).
    """

    def __init__(self, n_residues: int, hidden_dim: int = 256) -> None:
        """Initialise the flow model.

        Parameters
        ----------
        n_residues:
            Number of Cα atoms in the protein representation.
        hidden_dim:
            Number of neurons in each hidden layer.
        """
        super().__init__()
        self.n_residues = n_residues
        self.hidden_dim = hidden_dim

        input_dim = 3 * n_residues + 1  # flattened coords + scalar t
        output_dim = 3 * n_residues

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, coords: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the velocity field at conformation ``coords`` and time ``t``.

        Parameters
        ----------
        coords:
            Cα coordinates of shape ``(N, 3)`` or ``(B, N, 3)`` for a
            batched call.  The tensor is flattened along the last two
            dimensions before being fed to the MLP.
        t:
            Scalar or 1-D tensor containing the current integration
            time in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Velocity field of shape ``(N, 3)`` (unbatched) or
            ``(B, N, 3)`` (batched), matching the input shape.
        """
        batch_mode = coords.dim() == 3
        if batch_mode:
            B, N, _ = coords.shape
            coords_flat = coords.reshape(B, -1)        # (B, 3N)
            t_expand = t.reshape(B, 1) if t.dim() > 0 else t.expand(B, 1)
            x = torch.cat([coords_flat, t_expand], dim=-1)   # (B, 3N+1)
            velocity = self.net(x).reshape(B, N, 3)
        else:
            N = self.n_residues
            coords_flat = coords.reshape(-1)           # (3N,)
            t_scalar = t.reshape(1) if t.dim() == 0 else t[:1]
            x = torch.cat([coords_flat, t_scalar], dim=0)    # (3N+1,)
            velocity = self.net(x).reshape(N, 3)

        return velocity

    def integrate(
        self,
        source: torch.Tensor,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Euler-integrate the velocity field from ``t=0`` to ``t=1``.

        Parameters
        ----------
        source:
            Starting conformation of shape ``(N, 3)``.
        num_steps:
            Number of Euler integration steps.

        Returns
        -------
        torch.Tensor
            Trajectory tensor of shape ``(num_steps + 1, N, 3)``
            including both the source frame (``t=0``) and the final
            integrated frame (``t=1``).
        """
        device = source.device
        dt = 1.0 / num_steps

        frames: list[torch.Tensor] = [source]
        coords = source.clone()

        for step in range(num_steps):
            t_val = torch.tensor(step * dt, dtype=source.dtype, device=device)
            with torch.no_grad():
                velocity = self.forward(coords, t_val)
            coords = coords + dt * velocity
            frames.append(coords)

        return torch.stack(frames, dim=0)  # (num_steps+1, N, 3)
