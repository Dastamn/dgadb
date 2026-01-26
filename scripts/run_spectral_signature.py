#!/usr/bin/env python3
"""Run spectral signature analysis (Experiment 4c).

Computes full eigenvalue distributions and spectral metrics for clean and
anomalous graphs across all anomaly types. Generates per-dataset and
cross-dataset visualizations showing how different anomaly types shift
the spectral signature.
"""

import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import polars as pl
import typer

from dgadb.analysis.spectral import (
    compute_full_eigenvalues,
    compute_normalized_laplacian,
    compute_spectral_metrics,
    get_binary_adjacency,
)
from dgadb.analysis.visualization import (
    plot_concentration_at_one,
    plot_delta_mean_heatmap,
    plot_eigenvalue_histogram_comparison,
    plot_eigenvalue_kde_overlay,
    plot_frequency_band_bars,
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


def analyze_signature(
    dataset: str,
    output_dir: str,
    anom_types: list[str],
    anom_ratio: float = ANOMALY_RATIO,
    duration: str = DURATION,
) -> dict:
    """Analyze spectral signature for a dataset across all anomaly types.

    Args:
        dataset: Dataset name.
        output_dir: Directory to save results.
        anom_types: List of anomaly types to analyze.
        anom_ratio: Anomaly injection ratio for val/test.
        duration: Anomaly duration (small/medium/large).

    Returns:
        Dictionary with spectral signature results.
    """
    logger = logging.getLogger(f"spectral_signature.{dataset}")
    logger.info(f"Starting spectral signature analysis for {dataset}")

    os.makedirs(output_dir, exist_ok=True)

    loader = TemporalGraphLoaderNew()

    # Load clean graph and compute clean spectrum (train split)
    clean_tg = loader.load(dataset, create_if_not_found=True)
    clean_adj = get_binary_adjacency(clean_tg, clean_tg.train_mask)
    clean_L = compute_normalized_laplacian(clean_adj)

    logger.info(f"Computing clean eigenvalues for {dataset} (n={clean_L.shape[0]})...")
    clean_eigs = compute_full_eigenvalues(clean_L)
    clean_metrics = compute_spectral_metrics(clean_eigs)

    # Save clean eigenvalues
    np.save(os.path.join(output_dir, "clean_eigenvalues.npy"), clean_eigs)

    logger.info(
        f"Clean spectrum: {len(clean_eigs)} eigenvalues, "
        f"mean={clean_metrics['mean']:.4f}"
    )

    # Collect all condition data for plots
    eigenvalues_by_condition: dict[str, np.ndarray] = {"clean": clean_eigs}
    band_ratios: dict[str, dict[str, float]] = {
        "clean": {
            "low_freq_ratio": clean_metrics["low_freq_ratio"],
            "mid_freq_ratio": clean_metrics["mid_freq_ratio"],
            "high_freq_ratio": clean_metrics["high_freq_ratio"],
        }
    }

    # Results structure
    results = {
        "dataset": dataset,
        "anomaly_ratio": anom_ratio,
        "duration": duration,
        "clean": {
            "num_eigenvalues": len(clean_eigs),
            "is_full_decomposition": clean_L.shape[0] < 5000,
            "metrics": clean_metrics,
        },
        "anomalous": {},
    }

    # Compute spectrum for each anomaly type
    for anom_type in anom_types:
        logger.info(f"Processing {dataset}/{anom_type}...")

        anom_tg = loader.load(
            dataset,
            anom_type=anom_type,
            anom_val_ratio=anom_ratio,
            anom_test_ratio=anom_ratio,
            duration=duration,
            create_if_not_found=True,
        )

        anom_adj = get_binary_adjacency(anom_tg, anom_tg.test_mask)
        anom_L = compute_normalized_laplacian(anom_adj)

        anom_eigs = compute_full_eigenvalues(anom_L)
        anom_metrics = compute_spectral_metrics(anom_eigs)

        # Save anomalous eigenvalues
        np.save(os.path.join(output_dir, f"{anom_type}_eigenvalues.npy"), anom_eigs)

        # Compute deltas
        delta_metrics = {
            f"delta_{k}": anom_metrics[k] - clean_metrics[k] for k in clean_metrics
        }

        results["anomalous"][anom_type] = {
            "num_eigenvalues": len(anom_eigs),
            "is_full_decomposition": anom_L.shape[0] < 5000,
            "metrics": anom_metrics,
            "delta_metrics": delta_metrics,
        }

        eigenvalues_by_condition[anom_type] = anom_eigs
        band_ratios[anom_type] = {
            "low_freq_ratio": anom_metrics["low_freq_ratio"],
            "mid_freq_ratio": anom_metrics["mid_freq_ratio"],
            "high_freq_ratio": anom_metrics["high_freq_ratio"],
        }

        logger.info(f"  {anom_type}: delta_mean={delta_metrics['delta_mean']:+.4f}")

    # Save results JSON
    with open(os.path.join(output_dir, "spectral_signature.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Generate per-dataset plots
    plot_eigenvalue_kde_overlay(
        eigenvalues_by_condition,
        os.path.join(output_dir, "kde_overlay.png"),
        dataset,
    )

    for anom_type in anom_types:
        if anom_type in eigenvalues_by_condition:
            plot_eigenvalue_histogram_comparison(
                clean_eigs,
                eigenvalues_by_condition[anom_type],
                os.path.join(output_dir, f"histogram_{anom_type}.png"),
                dataset,
                anom_type,
            )

    plot_frequency_band_bars(
        band_ratios,
        os.path.join(output_dir, "frequency_bands.png"),
        dataset,
    )

    logger.info(f"Completed spectral signature analysis for {dataset}")
    return results


def analyze_signature_wrapper(args: tuple) -> dict:
    """Wrapper for multiprocessing."""
    dataset, output_dir, anom_types, anom_ratio, duration = args
    try:
        return analyze_signature(dataset, output_dir, anom_types, anom_ratio, duration)
    except Exception as e:
        logging.error(f"Failed to analyze signature for {dataset}: {e}", exc_info=True)
        return {"dataset": dataset, "error": str(e)}


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
    """Run spectral signature analysis for all datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    target_datasets = datasets if datasets else DATASETS
    target_anom_types = anom_types if anom_types else ANOMALY_TYPES

    logging.info(
        f"Analyzing spectral signatures for {len(target_datasets)} datasets "
        f"with {len(target_anom_types)} anomaly types"
    )

    # Prepare tasks
    tasks = [
        (
            ds,
            os.path.join(output_base, ds, "signature"),
            target_anom_types,
            anom_ratio,
            duration,
        )
        for ds in target_datasets
    ]

    # Run in parallel
    all_results = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for result in executor.map(analyze_signature_wrapper, tasks):
            if "error" not in result:
                all_results[result["dataset"]] = result

    # Cross-dataset plots
    plots_dir = os.path.join(output_base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Delta mean heatmap
    delta_mean_data: dict[str, dict[str, float]] = {}
    for ds, ds_result in all_results.items():
        delta_mean_data[ds] = {
            at: anom_result["delta_metrics"]["delta_mean"]
            for at, anom_result in ds_result.get("anomalous", {}).items()
        }

    if delta_mean_data:
        plot_delta_mean_heatmap(
            delta_mean_data,
            os.path.join(plots_dir, "delta_mean_heatmap.png"),
        )

    # Concentration at one (clean vs clique)
    concentration_data: dict[str, dict[str, float]] = {}
    for ds, ds_result in all_results.items():
        clean_conc = (
            ds_result.get("clean", {}).get("metrics", {}).get("concentration_at_1", 0.0)
        )
        clique_conc = (
            ds_result.get("anomalous", {})
            .get("clique", {})
            .get("metrics", {})
            .get("concentration_at_1", 0.0)
        )
        concentration_data[ds] = {"clean": clean_conc, "clique": clique_conc}

    if concentration_data:
        plot_concentration_at_one(
            concentration_data,
            os.path.join(plots_dir, "concentration_at_one.png"),
        )

    # Build summary CSV using polars
    records = []
    for ds, ds_result in all_results.items():
        # Clean row
        clean_m = ds_result.get("clean", {}).get("metrics", {})
        records.append(
            {
                "dataset": ds,
                "condition": "clean",
                "mean": clean_m.get("mean"),
                "std": clean_m.get("std"),
                "median": clean_m.get("median"),
                "skewness": clean_m.get("skewness"),
                "low_freq_pct": (clean_m.get("low_freq_ratio", 0) * 100),
                "mid_freq_pct": (clean_m.get("mid_freq_ratio", 0) * 100),
                "high_freq_pct": (clean_m.get("high_freq_ratio", 0) * 100),
                "spectral_gap": clean_m.get("spectral_gap"),
                "concentration_at_1": clean_m.get("concentration_at_1"),
                "delta_mean": None,
                "shift_direction": None,
            }
        )

        # Anomalous rows
        for at, anom_result in ds_result.get("anomalous", {}).items():
            anom_m = anom_result.get("metrics", {})
            delta_m = anom_result.get("delta_metrics", {})
            d_mean = delta_m.get("delta_mean", 0.0)
            records.append(
                {
                    "dataset": ds,
                    "condition": at,
                    "mean": anom_m.get("mean"),
                    "std": anom_m.get("std"),
                    "median": anom_m.get("median"),
                    "skewness": anom_m.get("skewness"),
                    "low_freq_pct": (anom_m.get("low_freq_ratio", 0) * 100),
                    "mid_freq_pct": (anom_m.get("mid_freq_ratio", 0) * 100),
                    "high_freq_pct": (anom_m.get("high_freq_ratio", 0) * 100),
                    "spectral_gap": anom_m.get("spectral_gap"),
                    "concentration_at_1": anom_m.get("concentration_at_1"),
                    "delta_mean": d_mean,
                    "shift_direction": "LEFT" if d_mean < 0 else "RIGHT",
                }
            )

    if records:
        df = pl.DataFrame(records)
        df.write_csv(os.path.join(output_base, "signature_summary.csv"))

    # Save summary JSON
    summary = {
        "datasets_analyzed": list(all_results.keys()),
        "anomaly_types": target_anom_types,
        "anomaly_ratio": anom_ratio,
        "duration": duration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "delta_mean": delta_mean_data,
    }
    with open(os.path.join(output_base, "signature_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(
        f"Spectral signature analysis complete. Results saved to {output_base}"
    )


if __name__ == "__main__":
    app()
