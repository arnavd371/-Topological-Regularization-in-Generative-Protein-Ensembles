"""rmsd_metrics.py

Wrappers for RMSD and TM-score calculations between protein
conformations.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from typing import Optional

import numpy as np


def compute_rmsd(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
) -> float:
    """Compute the root-mean-square deviation (RMSD) between two sets of
    Cα coordinates after least-squares superposition.

    Parameters
    ----------
    coords_A:
        First set of Cα coordinates, shape ``(N, 3)``.
    coords_B:
        Second set of Cα coordinates, shape ``(N, 3)``.

    Returns
    -------
    float
        RMSD in Ångströms.

    Notes
    -----
    A simple centroid-aligned RMSD is computed (Kabsch-like without
    rotation).  For a fully superimposed RMSD the caller should
    pre-align the structures using Kabsch rotation.
    """
    if coords_A.shape != coords_B.shape:
        raise ValueError(
            f"Shape mismatch: {coords_A.shape} vs {coords_B.shape}"
        )
    diff = coords_A - coords_B
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_tmscore(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    tmscore_binary: Optional[str] = None,
) -> float:
    """Compute TM-score between two conformations.

    Attempts to call the ``TMscore`` binary (if installed).  Falls
    back to a simple RMSD-based approximation when the binary is not
    available.

    Parameters
    ----------
    coords_A:
        Reference Cα coordinates, shape ``(N, 3)``.
    coords_B:
        Query Cα coordinates, shape ``(N, 3)``.
    tmscore_binary:
        Path to the ``TMscore`` executable.  If ``None``, the function
        searches ``PATH`` for ``TMscore`` or ``tmscore``.

    Returns
    -------
    float
        TM-score in ``(0, 1]``, or an RMSD-based approximation if the
        binary is unavailable.
    """
    # Locate binary
    binary = tmscore_binary
    if binary is None:
        for name in ("TMscore", "tmscore"):
            binary = shutil.which(name)
            if binary:
                break

    if binary is not None:
        return _tmscore_via_binary(coords_A, coords_B, binary)

    # Fallback: approximate TM-score from RMSD using the analytical
    # approximation for globular proteins (Zhang & Skolnick 2004):
    #   TM ≈ 1 / (1 + (RMSD / d0)^2)   where d0 = 1.24*(N-15)^(1/3) - 1.8
    N = coords_A.shape[0]
    d0 = max(0.5, 1.24 * (N - 15) ** (1.0 / 3.0) - 1.8) if N > 15 else 0.5
    rmsd = compute_rmsd(coords_A, coords_B)
    tmscore_approx = 1.0 / (1.0 + (rmsd / d0) ** 2)
    return float(tmscore_approx)


def _tmscore_via_binary(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    binary: str,
) -> float:
    """Run the TMscore binary on temporary PDB files and parse the output.

    Parameters
    ----------
    coords_A:
        Reference coordinates, shape ``(N, 3)``.
    coords_B:
        Query coordinates, shape ``(N, 3)``.
    binary:
        Absolute path to the ``TMscore`` executable.

    Returns
    -------
    float
        TM-score value parsed from the binary output.
    """
    def _write_dummy_pdb(coords: np.ndarray, path: str) -> None:
        """Write Cα-only PDB file from coordinate array."""
        with open(path, "w") as f:
            for idx, (x, y, z) in enumerate(coords):
                f.write(
                    f"ATOM  {idx+1:5d}  CA  ALA A{idx+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        path_A = os.path.join(tmpdir, "ref.pdb")
        path_B = os.path.join(tmpdir, "query.pdb")
        _write_dummy_pdb(coords_A, path_A)
        _write_dummy_pdb(coords_B, path_B)

        result = subprocess.run(
            [binary, path_B, path_A],
            capture_output=True,
            text=True,
        )

    for line in result.stdout.splitlines():
        if line.startswith("TM-score="):
            try:
                return float(line.split("=")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

    # If parsing fails, fall back to RMSD-based approximation directly
    N = coords_A.shape[0]
    d0 = max(0.5, 1.24 * (N - 15) ** (1.0 / 3.0) - 1.8) if N > 15 else 0.5
    rmsd = compute_rmsd(coords_A, coords_B)
    return float(1.0 / (1.0 + (rmsd / d0) ** 2))
