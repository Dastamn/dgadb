#!/usr/bin/env python3
"""Run spectral shift analysis (Experiment 4b).

Computes spectral shift (delta S_high) between clean and anomalous graphs
for all anomaly types. Results are saved to analysis-results/spectral/.
"""

import itertools
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
    compute_energy_ratio_curve,
    compute_normalized_laplacian,
    compute_s_high,
    get_binary_adjacency,
)
from dgadb.analysis.visualization import (
    plot_delta_s_high_grouped_bar,
    plot_delta_s_high_heatmap,
    plot_energy_ratio_curves,
)
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

ANOMALY_TYPES = ["random", "burst", "clique", "path", "bridge"]
ANOMALY_RATIO = 0.1
DURATION = "medium"


def analyze_shift(
    dataset: str,
    anom_type: str,
    output_dir: str,
    anom_ratio: float = ANOMALY_RATIO,
    duration: str = DURATION,
) -> dict:
    """Analyze spectral shift for a single (dataset, anomaly_type) pair.

    Args:
        dataset: Dataset name.
        anom_type: Anomaly type to inject.
        output_dir: Directory to save results.
        anom_ratio: Anomaly injection ratio for val/test.
        duration: Anomaly duration (small/medium/large).

    Returns:
        Dictionary with spectral shift metrics.
    """
    logger = logging.getLogger(f"spectral_shift.{dataset}.{anom_type}")
    logger.info(f"Analyzing spectral shift: {dataset} with {anom_type}")

    os.makedirs(output_dir, exist_ok=True)

    loader = TemporalGraphLoaderNew()

    # Load clean graph
    clean_tg = loader.load(dataset, create_if_not_found=True)

    # Load anomalous graph
    anom_tg = loader.load(
        dataset,
        anom_type=anom_type,
        anom_val_ratio=anom_ratio,
        anom_test_ratio=anom_ratio,
        duration=duration,
        create_if_not_found=True,
    )

    # Compute spectral properties for clean graph (train split)
    clean_adj = get_binary_adjacency(clean_tg, clean_tg.train_mask)
    clean_L = compute_normalized_laplacian(clean_adj)
    clean_signal = compute_degree_signal(clean_adj)
    clean_s_high = compute_s_high(clean_L, clean_signal)

    # Compute spectral properties for anomalous graph (test split includes anomalies)
    anom_adj = get_binary_adjacency(anom_tg, anom_tg.test_mask)
    anom_L = compute_normalized_laplacian(anom_adj)
    anom_signal = compute_degree_signal(anom_adj)
    anom_s_high = compute_s_high(anom_L, anom_signal)

    delta_s_high = anom_s_high - clean_s_high

    # Compute energy ratio curves (need eigenvectors)
    logger.info("Computing eigenvectors for energy ratio curves...")

    clean_eigenvalues, clean_eigenvectors = compute_eigenvalues(
        clean_L, return_eigenvectors=True
    )
    anom_eigenvalues, anom_eigenvectors = compute_eigenvalues(
        anom_L, return_eigenvectors=True
    )

    clean_energy_curve = compute_energy_ratio_curve(
        clean_eigenvalues, clean_eigenvectors, clean_signal
    )
    anom_energy_curve = compute_energy_ratio_curve(
        anom_eigenvalues, anom_eigenvectors, anom_signal
    )

    # Save energy ratio data
    np.savez(
        os.path.join(output_dir, "energy_curves.npz"),
        clean_eigenvalues=clean_energy_curve[0],
        clean_energy=clean_energy_curve[1],
        anom_eigenvalues=anom_energy_curve[0],
        anom_energy=anom_energy_curve[1],
    )

    # Compile results
    results = {
        "dataset": dataset,
        "anomaly_type": anom_type,
        "anomaly_ratio": anom_ratio,
        "duration": duration,
        "clean_s_high": clean_s_high,
        "anom_s_high": anom_s_high,
        "delta_s_high": delta_s_high,
        "clean_num_nodes": int(clean_adj.shape[0]),
        "clean_num_edges": int(clean_adj.nnz // 2),
        "anom_num_nodes": int(anom_adj.shape[0]),
        "anom_num_edges": int(anom_adj.nnz // 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(os.path.join(output_dir, "spectral_shift.json"), "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Completed {dataset}/{anom_type}: delta_S_high = {delta_s_high:.4f}")
    return results


def analyze_shift_wrapper(args: tuple) -> dict:
    """Wrapper for multiprocessing."""
    dataset, anom_type, output_dir, anom_ratio, duration = args
    try:
        return analyze_shift(dataset, anom_type, output_dir, anom_ratio, duration)
    except Exception as e:
        logging.error(f"Failed to analyze {dataset}/{anom_type}: {e}", exc_info=True)
        return {"dataset": dataset, "anomaly_type": anom_type, "error": str(e)}


@app.command()
def main(
    n_workers: int = typer.Option(8, help="Number of parallel workers"),
    datasets: list[str] = typer.Option(None, help="Specific datasets to analyze"),
    anom_types: list[str] = typer.Option(None, help="Specific anomaly types"),
    anom_ratio: float = typer.Option(ANOMALY_RATIO, help="Anomaly injection ratio"),
    duration: str = typer.Option(DURATION, help="Anomaly duration"),
    output_base: str = typer.Option(
        "analysis-results/spectral", help="Base output directory"
    ),
) -> None:
    """Run spectral shift analysis for all (dataset, anomaly_type) pairs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    target_datasets = datasets if datasets else DATASETS
    target_anom_types = anom_types if anom_types else ANOMALY_TYPES

    logging.info(
        f"Analyzing {len(target_datasets)} datasets x {len(target_anom_types)} anomaly types"
    )

    # Prepare tasks
    tasks = []
    for ds, at in itertools.product(target_datasets, target_anom_types):
        output_dir = os.path.join(output_base, ds, "anomalous", at)
        tasks.append((ds, at, output_dir, anom_ratio, duration))

    # Run in parallel
    all_results = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for result in executor.map(analyze_shift_wrapper, tasks):
            if "error" not in result:
                ds = result["dataset"]
                at = result["anomaly_type"]
                if ds not in all_results:
                    all_results[ds] = {}
                all_results[ds][at] = result

    # Create summary plots
    plots_dir = os.path.join(output_base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare delta S_high data for heatmap
    delta_s_high_data = {}
    for ds, anom_results in all_results.items():
        delta_s_high_data[ds] = {
            at: r["delta_s_high"] for at, r in anom_results.items()
        }

    if delta_s_high_data:
        plot_delta_s_high_heatmap(
            delta_s_high_data,
            os.path.join(plots_dir, "delta_s_high_heatmap.png"),
        )
        plot_delta_s_high_grouped_bar(
            delta_s_high_data,
            os.path.join(plots_dir, "delta_s_high_bar.png"),
        )

    # Create energy ratio plots per dataset
    for ds in target_datasets:
        # Load clean baseline curves
        clean_baseline_path = os.path.join(
            output_base, ds, "baseline", "eigenvalues.npy"
        )
        if not os.path.exists(clean_baseline_path):
            continue

        # Gather anomalous curves
        anomalous_curves = {}
        for at in target_anom_types:
            curve_path = os.path.join(
                output_base, ds, "anomalous", at, "energy_curves.npz"
            )
            if os.path.exists(curve_path):
                data = np.load(curve_path)
                anomalous_curves[at] = (data["anom_eigenvalues"], data["anom_energy"])

        if anomalous_curves:
            # Load clean curve from first anomalous result
            first_curve_path = os.path.join(
                output_base, ds, "anomalous", target_anom_types[0], "energy_curves.npz"
            )
            if os.path.exists(first_curve_path):
                clean_data = np.load(first_curve_path)
                clean_curve = (
                    clean_data["clean_eigenvalues"],
                    clean_data["clean_energy"],
                )

                plot_energy_ratio_curves(
                    clean_curve,
                    anomalous_curves,
                    os.path.join(plots_dir, f"energy_ratio_{ds}.png"),
                    dataset=ds,
                )

    # Save summary
    summary = {
        "datasets_analyzed": list(all_results.keys()),
        "anomaly_types": target_anom_types,
        "anomaly_ratio": anom_ratio,
        "duration": duration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "delta_s_high": delta_s_high_data,
    }
    with open(os.path.join(output_base, "shift_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Analysis complete. Results saved to {output_base}")


if __name__ == "__main__":
    app()
