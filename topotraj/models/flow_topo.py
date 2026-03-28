"""flow_topo.py

Flow-matching model with topological regularization.

``TopoFlowModel`` is a thin subclass of :class:`ConformerFlowModel`
that inherits the same architecture.  The topological loss is **not**
applied inside the model — it is computed externally in the training
loop (see :mod:`topotraj.training.train_topo`) by calling
:func:`topotraj.topology.topo_loss.trajectory_topological_loss` on
the trajectory produced by :meth:`integrate`.

This design keeps the model architecture clean and decoupled from the
regularization strategy.
"""

from __future__ import annotations

from topotraj.models.flow_baseline import ConformerFlowModel


class TopoFlowModel(ConformerFlowModel):
    """Flow-matching model with external topological regularization.

    Inherits all architecture and methods from
    :class:`~topotraj.models.flow_baseline.ConformerFlowModel`.  No
    new forward-pass logic is added; the topological loss is applied
    externally during training.

    Parameters
    ----------
    n_residues:
        Number of Cα atoms in the protein representation.
    hidden_dim:
        Width of each hidden layer (default: 256).
    """

    def __init__(self, n_residues: int, hidden_dim: int = 256) -> None:
        """Initialise the topological flow model.

        Parameters
        ----------
        n_residues:
            Number of Cα atoms in the protein representation.
        hidden_dim:
            Width of each hidden layer.
        """
        super().__init__(n_residues=n_residues, hidden_dim=hidden_dim)
