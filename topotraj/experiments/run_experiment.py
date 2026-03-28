"""run_experiment.py

Entry point for comparing the baseline flow model against the
topologically-regularized flow model (TopoFlow).

Usage
-----
With synthetic data (no PDB files required):

    python -m topotraj.experiments.run_experiment --synthetic

With real PDB pairs (in data/conformer_pairs/):

    python -m topotraj.experiments.run_experiment --config configs/base.yaml

Output
------
* Comparison table printed to stdout.
* All results saved to ``results/experiment_results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

from topotraj.data.conformation_loader import (
    generate_synthetic_conformer_pair,
    load_conformer_dataset,
)
from topotraj.training.train_baseline import train_baseline
from topotraj.training.train_topo import train_topo
from topotraj.evaluation.evaluate import evaluate_model


def _make_config(
    base_cfg: dict,
    train_pairs: list,
    test_pairs: list,
) -> dict:
    """Merge base config with runtime data references.

    Parameters
    ----------
    base_cfg:
        Dictionary loaded from ``base.yaml``.
    train_pairs:
        List of ``(coords_A, coords_B)`` pairs for training.
    test_pairs:
        List of ``(coords_A, coords_B)`` pairs for evaluation.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    cfg = dict(base_cfg)
    cfg["train_pairs"] = train_pairs
    cfg["test_pairs"] = test_pairs
    return cfg


def _print_table(
    baseline_results: dict,
    topo_results: dict,
) -> None:
    """Print a comparison table to stdout.

    Parameters
    ----------
    baseline_results:
        Evaluation metrics for the baseline model.
    topo_results:
        Evaluation metrics for the TopoFlow model.
    """
    col_w = 20
    row_fmt = f"{{:<{col_w}}} {{:<{col_w}}} {{:<{col_w}}}"
    sep = "-" * (col_w * 3 + 2)

    print(sep)
    print(row_fmt.format("Metric", "Baseline", "TopoFlow"))
    print(sep)

    def _fmt(metrics: dict, key_mean: str, key_std: str) -> str:
        return f"{metrics[key_mean]:.4f} ± {metrics[key_std]:.4f}"

    print(row_fmt.format(
        "Clash rate (↓)",
        _fmt(baseline_results, "clash_rate_mean", "clash_rate_std"),
        _fmt(topo_results, "clash_rate_mean", "clash_rate_std"),
    ))
    print(row_fmt.format(
        "Endpoint RMSD (↓)",
        _fmt(baseline_results, "endpoint_rmsd_mean", "endpoint_rmsd_std"),
        _fmt(topo_results, "endpoint_rmsd_mean", "endpoint_rmsd_std"),
    ))
    print(row_fmt.format(
        "Mean H1 persistence",
        _fmt(baseline_results, "mean_h1_persistence_mean", "mean_h1_persistence_std"),
        _fmt(topo_results, "mean_h1_persistence_mean", "mean_h1_persistence_std"),
    ))
    print(sep)


def run_experiment(config: dict) -> dict:
    """Run the full baseline-vs-TopoFlow comparison experiment.

    Trains both models on the same training pairs, evaluates both on
    the same test pairs, prints a comparison table, and saves results
    to ``output_dir/experiment_results.json``.

    Parameters
    ----------
    config:
        Configuration dictionary containing ``train_pairs``,
        ``test_pairs``, and all hyper-parameters.

    Returns
    -------
    dict
        Dictionary with keys ``"baseline"`` and ``"topo"``, each
        containing the evaluation metrics dict.
    """
    print("=== Training baseline model ===")
    baseline_model = train_baseline(config)

    print("\n=== Training TopoFlow model ===")
    topo_model = train_topo(config)

    test_pairs = config["test_pairs"]
    num_steps = int(config.get("num_integration_steps", 50))
    max_edge_length = float(config.get("max_edge_length", 8.0))
    clash_threshold = float(config.get("clash_threshold", 2.0))

    print("\n=== Evaluating baseline model ===")
    baseline_results = evaluate_model(
        baseline_model,
        test_pairs,
        num_steps=num_steps,
        max_edge_length=max_edge_length,
        clash_threshold=clash_threshold,
    )

    print("=== Evaluating TopoFlow model ===")
    topo_results = evaluate_model(
        topo_model,
        test_pairs,
        num_steps=num_steps,
        max_edge_length=max_edge_length,
        clash_threshold=clash_threshold,
    )

    print("\n=== Results ===")
    _print_table(baseline_results, topo_results)

    # Save to JSON
    output_dir = config.get("output_dir", "results/")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_results.json")
    results = {"baseline": baseline_results, "topo": topo_results}
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main() -> None:
    """Parse command-line arguments and run the experiment.

    Flags
    -----
    ``--synthetic`` : Use synthetic conformer pairs (no PDB files needed).
    ``--n-residues`` : Number of residues for synthetic proteins (default 50).
    ``--n-pairs`` : Number of synthetic pairs to generate (default 4).
    ``--config`` : Path to YAML config file (default: configs/base.yaml).
    """
    parser = argparse.ArgumentParser(
        description="Run baseline vs. TopoFlow conformer experiment."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real PDB files.",
    )
    parser.add_argument(
        "--n-residues",
        type=int,
        default=50,
        help="Residues per synthetic protein (default: 50).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=4,
        help="Number of synthetic conformer pairs (default: 4).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    # Load base config
    if os.path.isfile(args.config):
        with open(args.config) as f:
            base_cfg = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}. Using defaults.", file=sys.stderr)
        base_cfg = {}

    # Prepare data
    if args.synthetic:
        print(f"Generating {args.n_pairs} synthetic conformer pairs "
              f"({args.n_residues} residues each) ...")
        all_pairs = [
            generate_synthetic_conformer_pair(n_residues=args.n_residues)
            for _ in range(args.n_pairs)
        ]
    else:
        data_dir = base_cfg.get("data_dir", "data/conformer_pairs/")
        print(f"Loading conformer pairs from {data_dir} ...")
        all_pairs = load_conformer_dataset(data_dir)
        if not all_pairs:
            print("No conformer pairs found. Use --synthetic for smoke testing.",
                  file=sys.stderr)
            sys.exit(1)

    # Split 75/25 train/test
    split = max(1, int(0.75 * len(all_pairs)))
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:] if len(all_pairs) > split else all_pairs[:1]

    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    config = _make_config(base_cfg, train_pairs, test_pairs)

    # Reduce training steps for quick smoke test
    if args.synthetic and "training_steps" not in base_cfg:
        config["training_steps"] = 500

    run_experiment(config)


if __name__ == "__main__":
    main()
