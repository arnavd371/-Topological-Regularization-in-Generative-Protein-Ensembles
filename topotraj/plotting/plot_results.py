"""plot_results.py

Paper-quality figures for comparing the baseline and TopoFlow models.

All figures are saved at 300 DPI using a colorblind-safe palette
(Wong 2011 eight-colour palette).

Figures produced
----------------
1. Bar chart — clash rate (baseline vs. TopoFlow).
2. Line plot — mean H1 persistence across trajectory timesteps.
3. Example persistence diagram for a generated trajectory frame.
4. RMSD from start vs. timestep for an example trajectory.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from topotraj.topology.backbone_complex import coords_to_simplex_tree
from topotraj.topology.persistence import (
    compute_h1_persistence,
    total_h1_persistence,
)
from topotraj.topology.visualize_pd import plot_persistence_diagram
from topotraj.evaluation.rmsd_metrics import compute_rmsd

# Wong (2011) colorblind-safe palette
_COLORS = {
    "black":       "#000000",
    "orange":      "#E69F00",
    "sky_blue":    "#56B4E9",
    "green":       "#009E73",
    "yellow":      "#F0E442",
    "blue":        "#0072B2",
    "vermilion":   "#D55E00",
    "pink":        "#CC79A7",
}


def _compute_h1_per_frame(
    model: Any,
    source: np.ndarray,
    num_steps: int = 50,
    max_edge_length: float = 8.0,
) -> List[float]:
    """Compute total H1 persistence for each trajectory frame.

    Parameters
    ----------
    model:
        Trained flow model with an ``integrate`` method.
    source:
        Source Cα coordinates, shape ``(N, 3)``.
    num_steps:
        Number of integration steps.
    max_edge_length:
        Rips complex edge threshold (Å).

    Returns
    -------
    List[float]
        Total H1 persistence value for each frame in the trajectory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_tensor = torch.tensor(source, dtype=torch.float32, device=device)
    traj = model.integrate(source_tensor, num_steps=num_steps).cpu().numpy()
    return [
        total_h1_persistence(
            compute_h1_persistence(
                coords_to_simplex_tree(traj[t], max_edge_length=max_edge_length)
            )
        )
        for t in range(traj.shape[0])
    ]


def _compute_rmsd_per_frame(
    model: Any,
    source: np.ndarray,
    num_steps: int = 50,
) -> List[float]:
    """Compute RMSD from the source conformation for each trajectory frame.

    Parameters
    ----------
    model:
        Trained flow model with an ``integrate`` method.
    source:
        Source Cα coordinates, shape ``(N, 3)``.
    num_steps:
        Number of integration steps.

    Returns
    -------
    List[float]
        RMSD from source for each frame in the trajectory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_tensor = torch.tensor(source, dtype=torch.float32, device=device)
    traj = model.integrate(source_tensor, num_steps=num_steps).cpu().numpy()
    return [compute_rmsd(traj[t], source) for t in range(traj.shape[0])]


def plot_clash_rate_comparison(
    baseline_results: Dict[str, float],
    topo_results: Dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """Figure 1: Bar chart comparing clash rates of baseline and TopoFlow.

    Parameters
    ----------
    baseline_results:
        Evaluation metrics dict for the baseline model (must contain
        ``clash_rate_mean`` and ``clash_rate_std``).
    topo_results:
        Evaluation metrics dict for the TopoFlow model.
    save_path:
        Output file path.  If ``None``, ``plt.show()`` is called.
    """
    models = ["Baseline", "TopoFlow"]
    means = [
        baseline_results["clash_rate_mean"],
        topo_results["clash_rate_mean"],
    ]
    stds = [
        baseline_results["clash_rate_std"],
        topo_results["clash_rate_std"],
    ]
    colors = [_COLORS["blue"], _COLORS["vermilion"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        models, means, yerr=stds,
        color=colors, edgecolor="k", linewidth=0.8,
        capsize=6, width=0.5,
    )
    ax.set_ylabel("Steric Clash Rate (↓ better)")
    ax.set_title("Figure 1: Clash Rate Comparison")
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_h1_persistence_trajectory(
    baseline_model: Any,
    topo_model: Any,
    source: np.ndarray,
    num_steps: int = 50,
    max_edge_length: float = 8.0,
    save_path: Optional[str] = None,
) -> None:
    """Figure 2: H1 persistence across trajectory timesteps.

    Generates trajectories from both models and plots the total H1
    persistence at each frame.

    Parameters
    ----------
    baseline_model:
        Trained baseline flow model.
    topo_model:
        Trained TopoFlow model.
    source:
        Source Cα coordinates, shape ``(N, 3)``.
    num_steps:
        Number of integration steps.
    max_edge_length:
        Rips complex edge threshold (Å).
    save_path:
        Output file path.  If ``None``, ``plt.show()`` is called.
    """
    base_h1 = _compute_h1_per_frame(baseline_model, source, num_steps, max_edge_length)
    topo_h1 = _compute_h1_per_frame(topo_model, source, num_steps, max_edge_length)
    timesteps = np.linspace(0, 1, len(base_h1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timesteps, base_h1, color=_COLORS["blue"], label="Baseline", linewidth=1.5)
    ax.plot(timesteps, topo_h1, color=_COLORS["vermilion"], label="TopoFlow",
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("Integration time t")
    ax.set_ylabel("Total H1 Persistence")
    ax.set_title("Figure 2: H1 Persistence Across Trajectory")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_example_persistence_diagram(
    model: Any,
    source: np.ndarray,
    frame_index: int = 25,
    num_steps: int = 50,
    max_edge_length: float = 8.0,
    save_path: Optional[str] = None,
) -> None:
    """Figure 3: Persistence diagram for a mid-trajectory frame.

    Parameters
    ----------
    model:
        A trained flow model with an ``integrate`` method.
    source:
        Source Cα coordinates, shape ``(N, 3)``.
    frame_index:
        Which trajectory frame to visualise (default: 25).
    num_steps:
        Number of integration steps.
    max_edge_length:
        Rips complex edge threshold (Å).
    save_path:
        Output file path.  If ``None``, ``plt.show()`` is called.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_tensor = torch.tensor(source, dtype=torch.float32, device=device)

    traj = model.integrate(source_tensor, num_steps=num_steps).cpu().numpy()
    idx = min(frame_index, traj.shape[0] - 1)
    st = coords_to_simplex_tree(traj[idx], max_edge_length=max_edge_length)
    bars = compute_h1_persistence(st)

    plot_persistence_diagram(
        bars,
        title=f"Figure 3: H1 Persistence Diagram (frame {idx})",
        save_path=save_path,
    )


def plot_rmsd_vs_timestep(
    baseline_model: Any,
    topo_model: Any,
    source: np.ndarray,
    num_steps: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """Figure 4: RMSD from start conformation vs. integration timestep.

    Parameters
    ----------
    baseline_model:
        Trained baseline flow model.
    topo_model:
        Trained TopoFlow model.
    source:
        Source Cα coordinates, shape ``(N, 3)``.
    num_steps:
        Number of integration steps.
    save_path:
        Output file path.  If ``None``, ``plt.show()`` is called.
    """
    base_rmsd = _compute_rmsd_per_frame(baseline_model, source, num_steps)
    topo_rmsd = _compute_rmsd_per_frame(topo_model, source, num_steps)
    timesteps = np.linspace(0, 1, len(base_rmsd))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timesteps, base_rmsd, color=_COLORS["blue"], label="Baseline", linewidth=1.5)
    ax.plot(timesteps, topo_rmsd, color=_COLORS["vermilion"], label="TopoFlow",
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("Integration time t")
    ax.set_ylabel("RMSD from source (Å)")
    ax.set_title("Figure 4: RMSD from Start vs. Timestep")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def generate_all_figures(
    baseline_model: Any,
    topo_model: Any,
    baseline_results: Dict[str, float],
    topo_results: Dict[str, float],
    source: np.ndarray,
    output_dir: str = "results/figures/",
    num_steps: int = 50,
    max_edge_length: float = 8.0,
) -> None:
    """Generate and save all four paper figures.

    Parameters
    ----------
    baseline_model:
        Trained baseline flow model.
    topo_model:
        Trained TopoFlow model.
    baseline_results:
        Evaluation metrics dict for the baseline model.
    topo_results:
        Evaluation metrics dict for the TopoFlow model.
    source:
        Example source Cα coordinates for trajectory visualisations.
    output_dir:
        Directory where figures are saved.
    num_steps:
        Integration steps for trajectory generation.
    max_edge_length:
        Rips complex edge threshold (Å).
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_clash_rate_comparison(
        baseline_results,
        topo_results,
        save_path=os.path.join(output_dir, "fig1_clash_rate.png"),
    )
    print(f"Figure 1 saved to {output_dir}fig1_clash_rate.png")

    plot_h1_persistence_trajectory(
        baseline_model,
        topo_model,
        source,
        num_steps=num_steps,
        max_edge_length=max_edge_length,
        save_path=os.path.join(output_dir, "fig2_h1_persistence.png"),
    )
    print(f"Figure 2 saved to {output_dir}fig2_h1_persistence.png")

    plot_example_persistence_diagram(
        topo_model,
        source,
        frame_index=num_steps // 2,
        num_steps=num_steps,
        max_edge_length=max_edge_length,
        save_path=os.path.join(output_dir, "fig3_persistence_diagram.png"),
    )
    print(f"Figure 3 saved to {output_dir}fig3_persistence_diagram.png")

    plot_rmsd_vs_timestep(
        baseline_model,
        topo_model,
        source,
        num_steps=num_steps,
        save_path=os.path.join(output_dir, "fig4_rmsd_vs_time.png"),
    )
    print(f"Figure 4 saved to {output_dir}fig4_rmsd_vs_time.png")
