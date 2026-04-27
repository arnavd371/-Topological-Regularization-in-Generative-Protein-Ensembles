# Topological Regularization in Generative Protein Ensembles (TopoTraj)

This repository implements a flow-matching framework for protein conformational interpolation, with an optional **topological regularization** term that discourages non-physical loop artifacts in generated trajectories.

The core comparison is:
- **Baseline model**: standard flow matching for conformer interpolation.
- **TopoFlow model**: same architecture, trained with an additional persistent-homology-based loss.

---

## 1) Project Goal

Protein conformations evolve along trajectories that should remain physically plausible. Purely geometric interpolation can introduce unrealistic backbone behaviors (e.g., spurious loops or clashes).  

This project investigates whether **persistent homology (H1 features)** can be used as a regularizer during training to produce cleaner generative trajectories.

---

## 2) Method Summary

### Baseline (Flow Matching)
- Learn a velocity field \(v(x_t, t)\) that transports source coordinates toward target coordinates.
- Training target velocity is derived from linear interpolation between paired conformers.

### Topological Regularization (TopoFlow)
- During training, integrate short trajectories from source conformations.
- Compute H1 persistence from Vietoris–Rips complexes of intermediate frames.
- Penalize significant loop birth edges via a differentiable surrogate loss.
- Optimize:
  - `total_loss = flow_loss + topo_loss`

---

## 3) Repository Structure

```text
configs/
  base.yaml                       # default hyperparameters and paths

topotraj/
  data/
    conformation_loader.py        # PDB pair loading + synthetic pair generation
    interpolate.py                # interpolation utilities
  models/
    flow_baseline.py              # ConformerFlowModel
    flow_topo.py                  # TopoFlowModel (architecture wrapper)
  topology/
    backbone_complex.py           # Cα extraction + Rips complex utilities
    persistence.py                # H1 persistence extraction/summaries
    topo_loss.py                  # differentiable topological loss
    visualize_pd.py               # persistence diagram/barcode plotting
  training/
    train_baseline.py             # baseline training loop
    train_topo.py                 # topologically-regularized training loop
  evaluation/
    steric_clash.py               # clash metrics
    rmsd_metrics.py               # RMSD and TM-score helper
    evaluate.py                   # evaluation runner
  plotting/
    plot_results.py               # paper-style figure generation
  experiments/
    run_experiment.py             # end-to-end baseline vs TopoFlow pipeline
```

---

## 4) Installation

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- GUDHI
- BioPython
- NumPy / SciPy
- Matplotlib
- PyYAML

---

## 5) Data Format

The loader expects paired files in:
`data/conformer_pairs/`

Naming convention:
- `<protein_id>_stateA.pdb`
- `<protein_id>_stateB.pdb`

Only proteins with both files are used. Coordinates are Cα-only and truncated to matching length.

For quick testing, synthetic pairs can be generated automatically (no PDB files required).

---

## 6) Configuration

Default config: `configs/base.yaml`

Important parameters:
- `training_steps`
- `flow_lr`
- `flow_hidden_dim`
- `num_interpolation_steps`
- `num_integration_steps`
- `topo_loss_weight`
- `max_edge_length`
- `persistence_threshold`
- `clash_threshold`
- `data_dir`
- `output_dir`

---

## 7) Running Experiments

### A) Smoke test with synthetic data
```bash
python -m topotraj.experiments.run_experiment --synthetic
```

Optional synthetic controls:
```bash
python -m topotraj.experiments.run_experiment --synthetic --n-residues 60 --n-pairs 8
```

### B) Run with real PDB conformer pairs
```bash
python -m topotraj.experiments.run_experiment --config configs/base.yaml
```

---

## 8) Outputs

The experiment:
- Trains baseline and TopoFlow models.
- Evaluates both using:
  - Clash rate (lower is better)
  - Endpoint RMSD (lower is better)
  - Mean H1 persistence
- Prints a comparison table.
- Saves JSON results to:
  - `results/experiment_results.json` (or configured `output_dir`)

The plotting module can generate paper-quality figures such as:
- clash-rate bar chart,
- H1 persistence over trajectory time,
- persistence diagram,
- RMSD-vs-time trajectory profile.

---

## 9) Interpreting Results

Typical success pattern for TopoFlow:
- Reduced clash rate,
- Similar or improved endpoint RMSD,
- Reduced H1 persistence in intermediate frames.

This indicates trajectories that are not only accurate at endpoints, but also more topologically and physically plausible along the full path.

---

## 10) How to Turn This Into a Research Paper

Use this project as a reproducible baseline-vs-regularized study.

### A) Paper framing (core narrative)
1. **Problem**: Generative conformational trajectories may contain non-physical intermediate states.
2. **Hypothesis**: Topological constraints (H1 persistence) reduce such artifacts.
3. **Approach**: Add differentiable topological regularization to flow matching.
4. **Evidence**: Compare baseline vs TopoFlow on physical and geometric metrics.

### B) Suggested paper structure
1. **Title**
   - Example direction: topological regularization for protein trajectory generation.
2. **Abstract**
   - Motivation → method → key quantitative gains → significance.
3. **Introduction**
   - Why intermediate-state realism matters.
   - Gaps in purely geometric generative methods.
4. **Related Work**
   - Protein generative modeling, flow matching, TDA in structural biology.
5. **Method**
   - Baseline flow objective.
   - Persistent homology setup (Rips, H1 bars, thresholds).
   - Surrogate gradient edge-based topological loss.
6. **Experimental Setup**
   - Dataset construction (stateA/stateB pairs).
   - Hyperparameters and training protocol.
   - Metrics and evaluation pipeline.
7. **Results**
   - Main comparison table (mean ± std).
   - Figures from `topotraj/plotting/plot_results.py`.
8. **Ablations**
   - Vary `topo_loss_weight`, `persistence_threshold`, `max_edge_length`.
   - Study tradeoff between endpoint fit and topological regularity.
9. **Discussion**
   - Biological plausibility, failure modes, scalability limits.
10. **Conclusion**
   - Summary and future work (e.g., all-atom models, larger datasets, energy-aware losses).

### C) Minimal experiment set for publication-ready evidence
- Baseline vs TopoFlow on the same train/test split.
- Multiple random seeds.
- Report means/stds for all metrics.
- Include statistical significance tests when sample size permits.
- Include qualitative trajectory case studies.

### D) Figure plan
- **Fig 1**: clash-rate comparison.
- **Fig 2**: H1 persistence over trajectory time.
- **Fig 3**: representative persistence diagrams.
- **Fig 4**: RMSD-vs-time profile.
- Optional: sensitivity plots for topo hyperparameters.

### E) Reproducibility checklist
- Fixed config file + random seeds.
- Exact dependency versions.
- Clear data split protocol.
- Public scripts for train/eval/plot.
- Raw result JSON files archived with paper artifacts.

### F) Limitations to acknowledge
- Cα-only representation.
- Approximate differentiable topology surrogate.
- Dataset scale and domain diversity.
- Potential compute overhead from topology calculations.

---

## 11) Practical Next Steps

To make this paper-ready quickly:
1. Add a curated real conformer benchmark set.
2. Run multi-seed experiments.
3. Export all standard figures/tables.
4. Add an `experiments/` manifest documenting each run.
5. Draft manuscript sections directly from this README structure.

---

## 12) Citation / Attribution

If you use this repository in academic work, please cite your paper/preprint and include a link to this repository.
