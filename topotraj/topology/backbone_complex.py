"""backbone_complex.py

Utilities for extracting Cα coordinates from PDB files and building
Vietoris-Rips simplicial complexes used for persistent-homology
computation.
"""

from __future__ import annotations

import numpy as np
import gudhi

from Bio.PDB import PDBParser


def extract_ca_coords(pdb_path: str) -> np.ndarray:
    """Parse a PDB file and return Cα coordinates for all residues.

    Parameters
    ----------
    pdb_path:
        Absolute or relative path to the PDB file.

    Returns
    -------
    np.ndarray
        Array of shape ``(N_residues, 3)`` containing the x, y, z
        coordinates (in Ångströms) of every Cα atom found in the
        structure, in chain/residue order.

    Notes
    -----
    Only the first model in the PDB file is used.  Residues that do
    not contain a ``CA`` atom (e.g. hetero-atoms) are silently skipped.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())

    ca_coords: list[list[float]] = []
    for chain in model.get_chains():
        for residue in chain.get_residues():
            if "CA" in residue:
                atom = residue["CA"]
                ca_coords.append(list(atom.get_vector()))

    return np.array(ca_coords, dtype=np.float64)


def build_rips_complex(
    coords: np.ndarray,
    max_edge_length: float = 8.0,
) -> gudhi.RipsComplex:
    """Build a Vietoris-Rips complex on Cα coordinates.

    Parameters
    ----------
    coords:
        Array of shape ``(N, 3)`` containing Cα coordinates in Ångströms.
    max_edge_length:
        Maximum edge length (in Ångströms) for the Rips complex.
        The default of 8.0 Å corresponds to a typical Cα–Cα contact
        threshold used in structural biology.

    Returns
    -------
    gudhi.RipsComplex
        An unfiltered ``RipsComplex`` object.  Call
        :func:`coords_to_simplex_tree` to obtain a ``SimplexTree``
        ready for persistent-homology computation.
    """
    rips = gudhi.RipsComplex(points=coords, max_edge_length=max_edge_length)
    return rips


def coords_to_simplex_tree(
    coords: np.ndarray,
    max_edge_length: float = 8.0,
    max_dimension: int = 2,
) -> gudhi.SimplexTree:
    """Build a Rips complex and return a ``SimplexTree`` expanded to
    *max_dimension*.

    A ``SimplexTree`` expanded to dimension 2 (i.e. containing
    triangles / 2-simplices) is required to compute H1 homology:
    without 2-simplices the Rips filtration has no triangles to fill
    in 1-cycles, so the persistence computation cannot detect loop
    deaths.

    Parameters
    ----------
    coords:
        Array of shape ``(N, 3)`` containing Cα coordinates.
    max_edge_length:
        Maximum edge length for the Rips complex (Ångströms).
    max_dimension:
        Dimension to which the ``SimplexTree`` is expanded.  Must be
        at least 2 to capture H1 features.

    Returns
    -------
    gudhi.SimplexTree
        A ``SimplexTree`` containing all simplices up to
        *max_dimension* within the Rips filtration.
    """
    rips = build_rips_complex(coords, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree
