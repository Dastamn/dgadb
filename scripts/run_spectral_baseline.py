#!/usr/bin/env python3
"""Run spectral baseline analysis (Experiment 4a).

Computes spectral properties (eigenvalues, S_high, spectral density) for
clean versions of all datasets. Results are saved to analysis-results/spectral/.
"""

import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import typer

from dgadb.analysis.spectral import (
    compute_degree_signal,
    compute_eigenvalues,
    compute_normalized_laplacian,
    compute_s_high,
    compute_spectral_density,
    get_binary_adjacency,
)
from dgadb.analysis.visualization import plot_eigenvalue_violin, plot_spectral_density
from dgadb.storage.temporal_graph import TemporalGraphLoaderNew

app = typer.Typer()

DATASETS = [
    "bitcoin-alpha",
    "bitcoin-otc",
    "as-topology",
    "digg-homo",
    "email-dnc",
    "uc-social",
]


def analyze_dataset(dataset: str, output_dir: str) -> dict:
    """Analyze spectral properties of a single dataset.

    Args:
        dataset: Dataset name to analyze.
        output_dir: Directory to save results.

    Returns:
        Dictionary with spectral metrics.
    """
    logger = logging.getLogger(f"spectral.{dataset}")
    logger.info(f"Starting spectral analysis for {dataset}")

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    loader = TemporalGraphLoaderNew()
    tg = loader.load(dataset, create_if_not_found=True)

    logger.info(f"Loaded {dataset}: {tg.num_nodes} nodes, {tg.num_edges} edges")

    # Get binary adjacency for train split
    adj = get_binary_adjacency(tg, tg.train_mask)
    num_unique_edges = adj.nnz // 2  # Divide by 2 since we made it undirected

    logger.info(
        f"Binary adjacency: {adj.shape[0]} nodes, {num_unique_edges} unique edges"
    )

    # Compute Laplacian
    L = compute_normalized_laplacian(adj)

    # Compute eigenvalues
    logger.info("Computing eigenvalues...")
    eigenvalues = compute_eigenvalues(L)

    # Compute degree signal and S_high
    signal = compute_degree_signal(adj)
    s_high = compute_s_high(L, signal)

    # Compute spectral density
    density = compute_spectral_density(eigenvalues)

    # Save eigenvalues
    np.save(os.path.join(output_dir, "eigenvalues.npy"), eigenvalues)

    # Compile results
    results = {
        "dataset": dataset,
        "graph_type": "clean",
        "num_nodes": int(tg.num_nodes),
        "num_edges": int(tg.num_edges),
        "num_unique_edges": int(num_unique_edges),
        "eigenvalue_stats": {
            "num_computed": len(eigenvalues),
            "min": float(eigenvalues.min()) if len(eigenvalues) > 0 else None,
            "max": float(eigenvalues.max()) if len(eigenvalues) > 0 else None,
            "mean": float(eigenvalues.mean()) if len(eigenvalues) > 0 else None,
            "std": float(eigenvalues.std()) if len(eigenvalues) > 0 else None,
        },
        "s_high": s_high,
        "spectral_density": density,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save metrics
    with open(os.path.join(output_dir, "spectral_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Generate density plot
    plot_spectral_density(
        eigenvalues,
        os.path.join(output_dir, "density.png"),
        dataset,
    )

    logger.info(f"Completed {dataset}: S_high = {s_high:.4f}")
    return results


def analyze_dataset_wrapper(args: tuple[str, str]) -> dict:
    """Wrapper for multiprocessing."""
    dataset, output_dir = args
    try:
        return analyze_dataset(dataset, output_dir)
    except Exception as e:
        logging.error(f"Failed to analyze {dataset}: {e}", exc_info=True)
        return {"dataset": dataset, "error": str(e)}


@app.command()
def main(
    n_workers: int = typer.Option(8, help="Number of parallel workers"),
    datasets: list[str] = typer.Option(None, help="Specific datasets to analyze"),
    output_base: str = typer.Option(
        "analysis-results/spectral", help="Base output directory"
    ),
) -> None:
    """Run spectral baseline analysis on all datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    target_datasets = datasets if datasets else DATASETS
    logging.info(f"Analyzing {len(target_datasets)} datasets with {n_workers} workers")

    # Prepare tasks
    tasks = [(ds, os.path.join(output_base, ds, "baseline")) for ds in target_datasets]

    # Run in parallel
    all_results = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for result in executor.map(analyze_dataset_wrapper, tasks):
            if "error" not in result:
                all_results[result["dataset"]] = result

    # Aggregate eigenvalues for cross-dataset comparison
    eigenvalue_data = {}
    for ds in target_datasets:
        eigen_path = os.path.join(output_base, ds, "baseline", "eigenvalues.npy")
        if os.path.exists(eigen_path):
            eigenvalue_data[ds] = np.load(eigen_path)

    # Create cross-dataset plots
    plots_dir = os.path.join(output_base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if eigenvalue_data:
        plot_eigenvalue_violin(
            eigenvalue_data,
            os.path.join(plots_dir, "eigenvalue_violin.png"),
        )

    # Save summary
    summary = {
        "datasets_analyzed": list(all_results.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": {
            ds: {k: v for k, v in r.items() if k != "spectral_density"}
            for ds, r in all_results.items()
        },
    }
    with open(os.path.join(output_base, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Analysis complete. Results saved to {output_base}")


if __name__ == "__main__":
    app()
