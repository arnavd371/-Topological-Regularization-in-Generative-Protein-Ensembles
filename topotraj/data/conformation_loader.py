"""conformation_loader.py

Utilities for loading paired protein conformations from PDB files and
for generating synthetic conformer pairs for smoke testing.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np

from topotraj.topology.backbone_complex import extract_ca_coords


def load_conformer_pair(
    pdb_path_A: str,
    pdb_path_B: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and align Cα coordinates from two PDB files.

    Both structures are truncated to the length of the shorter one so
    that the returned arrays have the same number of residues and can
    be used for interpolation and flow-matching training.

    Parameters
    ----------
    pdb_path_A:
        Path to the first (source) PDB file.
    pdb_path_B:
        Path to the second (target) PDB file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(coords_A, coords_B)`` each of shape ``(N, 3)`` where *N* is
        the length of the shorter structure.
    """
    coords_A = extract_ca_coords(pdb_path_A)
    coords_B = extract_ca_coords(pdb_path_B)

    # Align by truncating to the shorter sequence
    n = min(len(coords_A), len(coords_B))
    return coords_A[:n], coords_B[:n]


def load_conformer_dataset(
    data_dir: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load all paired conformer PDB files from a directory.

    Files must follow the naming convention
    ``<proteinX>_stateA.pdb`` / ``<proteinX>_stateB.pdb``.  All
    proteins for which both states are present are loaded.

    Parameters
    ----------
    data_dir:
        Directory containing the PDB files.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of ``(coords_A, coords_B)`` pairs, one per protein found.
    """
    pattern = re.compile(r"^(.+)_stateA\.pdb$")
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for filename in sorted(os.listdir(data_dir)):
        match = pattern.match(filename)
        if match:
            protein_id = match.group(1)
            path_A = os.path.join(data_dir, filename)
            path_B = os.path.join(data_dir, f"{protein_id}_stateB.pdb")
            if os.path.isfile(path_B):
                coords_A, coords_B = load_conformer_pair(path_A, path_B)
                pairs.append((coords_A, coords_B))

    return pairs


def generate_synthetic_conformer_pair(
    n_residues: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a physically plausible synthetic conformer pair.

    The source conformation is built as a 3-D random walk with step
    size 3.8 Å (the canonical Cα–Cα bond length).  The target
    conformation is obtained by adding independent Gaussian noise
    (σ = 2.0 Å) to each coordinate, mimicking a small conformational
    change.

    Parameters
    ----------
    n_residues:
        Number of residues (Cα atoms) in the synthetic protein.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(source, target)`` each of shape ``(n_residues, 3)``.
    """
    CA_BOND_LENGTH = 3.8   # Å — standard Cα–Cα distance along backbone
    NOISE_STD = 2.0        # Å — Gaussian perturbation for target state

    rng = np.random.default_rng()

    # Build source as a random walk
    steps = rng.normal(size=(n_residues - 1, 3))
    steps = steps / np.linalg.norm(steps, axis=1, keepdims=True) * CA_BOND_LENGTH
    source = np.zeros((n_residues, 3), dtype=np.float64)
    for i in range(1, n_residues):
        source[i] = source[i - 1] + steps[i - 1]

    # Target: perturb source with Gaussian noise
    target = source + rng.normal(scale=NOISE_STD, size=source.shape)

    return source, target
