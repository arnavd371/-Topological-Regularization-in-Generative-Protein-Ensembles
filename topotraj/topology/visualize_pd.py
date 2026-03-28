"""visualize_pd.py

Plotting utilities for persistence diagrams and barcodes, useful for
debugging and inspecting H1 topological features of backbone
conformations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_persistence_diagram(
    bars: List[Tuple[float, float]],
    title: str = "H1 Persistence Diagram",
    save_path: Optional[str] = None,
) -> None:
    """Plot a birth–death persistence diagram for H1 features.

    Points are coloured by persistence (``death - birth``) using a
    perceptually uniform colormap; a colorbar is included.  A diagonal
    reference line ``birth == death`` (zero-persistence boundary) is
    drawn for orientation.

    Parameters
    ----------
    bars:
        List of ``(birth, death)`` tuples representing H1 features.
    title:
        Title displayed above the plot.
    save_path:
        If provided, the figure is saved to this path at 300 DPI.
        Otherwise ``plt.show()`` is called.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    if bars:
        births = np.array([b for b, _ in bars])
        deaths = np.array([d for _, d in bars])
        persistence = deaths - births

        sc = ax.scatter(
            births,
            deaths,
            c=persistence,
            cmap="viridis",
            s=50,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )
        fig.colorbar(sc, ax=ax, label="Persistence (death − birth)")

        # Diagonal reference line
        all_vals = np.concatenate([births, deaths])
        vmin, vmax = all_vals.min() - 0.5, all_vals.max() + 0.5
    else:
        vmin, vmax = 0.0, 1.0

    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, label="birth = death")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Birth (Å)")
    ax.set_ylabel("Death (Å)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_persistence_barcode(
    bars: List[Tuple[float, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot a horizontal persistence barcode for H1 features.

    Bars are sorted by birth time (ascending) so that features
    entering the filtration earlier appear at the bottom.

    Parameters
    ----------
    bars:
        List of ``(birth, death)`` tuples representing H1 features.
    save_path:
        If provided, the figure is saved to this path at 300 DPI.
        Otherwise ``plt.show()`` is called.
    """
    # Sort bars by birth time
    sorted_bars = sorted(bars, key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_bars) * 0.4 + 1)))

    for idx, (birth, death) in enumerate(sorted_bars):
        ax.hlines(idx, birth, death, colors="#0072B2", linewidth=2)

    ax.set_xlabel("Filtration value (Å)")
    ax.set_ylabel("H1 feature index")
    ax.set_title("H1 Persistence Barcode")
    ax.set_yticks(range(len(sorted_bars)))
    ax.set_yticklabels([f"H1_{i}" for i in range(len(sorted_bars))], fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
