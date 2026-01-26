#!/usr/bin/env python3
"""Run multigraph analysis (Experiment 5).

Computes edge multiplicity statistics for all datasets and correlates
with DTDG/CTDG method performance. Results are saved to analysis-results/multigraph/.
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

from dgadb.analysis.multigraph import compute_multiplicity_stats
from dgadb.analysis.results_loader import load_experiment_results
from dgadb.analysis.visualization import (
    plot_correlation_scatter,
    plot_multiplicity_distribution,
    plot_multiplicity_vs_performance,
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

# Method categorization
DTDG_METHODS = ["strgnn", "taddy", "gcn", "gat", "graphsage"]
CTDG_METHODS = ["sad", "slade"]


def analyze_multiplicity(dataset: str, output_dir: str) -> dict:
    """Analyze edge multiplicity for a single dataset.

    Args:
        dataset: Dataset name.
        output_dir: Directory to save results.

    Returns:
        Dictionary with multiplicity statistics.
    """
    logger = logging.getLogger(f"multigraph.{dataset}")
    logger.info(f"Analyzing multiplicity for {dataset}")

    os.makedirs(output_dir, exist_ok=True)

    loader = TemporalGraphLoaderNew()
    tg = loader.load(dataset, create_if_not_found=True)

    # Compute multiplicity stats for full graph
    stats = compute_multiplicity_stats(tg)

    # Also compute for train split
    train_stats = compute_multiplicity_stats(tg, tg.train_mask)

    results = {
        "dataset": dataset,
        "full_graph": stats.to_dict(),
        "train_split": train_stats.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save results
    with open(os.path.join(output_dir, "multiplicity_stats.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plot distribution
    plot_multiplicity_distribution(
        stats.multiplicity_distribution,
        os.path.join(output_dir, "distribution.png"),
        dataset,
    )

    logger.info(
        f"Completed {dataset}: compression_ratio = {stats.compression_ratio:.3f}, "
        f"pct_repeated = {stats.pct_repeated:.1f}%"
    )
    return results


def analyze_multiplicity_wrapper(args: tuple[str, str]) -> dict:
    """Wrapper for multiprocessing."""
    dataset, output_dir = args
    try:
        return analyze_multiplicity(dataset, output_dir)
    except Exception as e:
        logging.error(f"Failed to analyze {dataset}: {e}", exc_info=True)
        return {"dataset": dataset, "error": str(e)}


def correlate_with_performance(
    multiplicity_results: dict[str, dict],
    experiment_dir: str,
    output_dir: str,
) -> dict:
    """Correlate multiplicity with method performance.

    Args:
        multiplicity_results: Dict mapping dataset to multiplicity stats.
        experiment_dir: Directory containing experiment results.
        output_dir: Directory to save correlation results.

    Returns:
        Dictionary with correlation analysis.
    """
    logger = logging.getLogger("multigraph.correlation")
    logger.info("Correlating multiplicity with performance...")

    os.makedirs(output_dir, exist_ok=True)

    # Load benchmark results
    performance_df = load_experiment_results(experiment_dir)

    if performance_df.is_empty():
        logger.warning("No benchmark results found")
        return {"error": "No benchmark results found"}

    # Add method type categorization
    performance_df = performance_df.with_columns(
        pl.when(pl.col("method").is_in(CTDG_METHODS))
        .then(pl.lit("CTDG"))
        .when(pl.col("method").is_in(DTDG_METHODS))
        .then(pl.lit("DTDG"))
        .otherwise(pl.lit("Other"))
        .alias("method_type")
    )

    # Create multiplicity DataFrame
    mult_records = []
    for dataset, result in multiplicity_results.items():
        if "error" in result:
            continue
        stats = result.get("full_graph", {})
        mult_records.append(
            {
                "dataset": dataset,
                "compression_ratio": stats.get("compression_ratio", 1.0),
                "pct_repeated": stats.get("pct_repeated", 0.0),
                "mean_multiplicity": stats.get("mean_multiplicity", 1.0),
            }
        )

    if not mult_records:
        logger.warning("No multiplicity data available")
        return {"error": "No multiplicity data available"}

    multiplicity_df = pl.DataFrame(mult_records)

    # Generate correlation plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_multiplicity_vs_performance(
        multiplicity_df,
        performance_df,
        os.path.join(plots_dir, "multiplicity_vs_performance.png"),
    )

    # Compute CTDG advantage vs compression ratio
    # Group by dataset and method_type, compute mean performance
    summary = (
        performance_df.filter(pl.col("roc_auc").is_not_null())
        .group_by(["dataset", "method_type"])
        .agg(pl.col("roc_auc").mean().alias("mean_roc_auc"))
    )

    # Pivot to get CTDG and DTDG columns
    ctdg_perf = (
        summary.filter(pl.col("method_type") == "CTDG")
        .select(["dataset", "mean_roc_auc"])
        .rename({"mean_roc_auc": "ctdg_auc"})
    )
    dtdg_perf = (
        summary.filter(pl.col("method_type") == "DTDG")
        .select(["dataset", "mean_roc_auc"])
        .rename({"mean_roc_auc": "dtdg_auc"})
    )

    merged = ctdg_perf.join(dtdg_perf, on="dataset", how="inner")
    merged = merged.with_columns(
        (pl.col("ctdg_auc") - pl.col("dtdg_auc")).alias("ctdg_advantage")
    )
    merged = merged.join(multiplicity_df, on="dataset", how="inner")

    if not merged.is_empty():
        # Plot compression ratio vs CTDG advantage
        plot_correlation_scatter(
            merged["compression_ratio"].to_numpy(),
            merged["ctdg_advantage"].to_numpy(),
            merged["dataset"].to_list(),
            os.path.join(plots_dir, "compression_vs_ctdg_advantage.png"),
            xlabel="Compression Ratio",
            ylabel="CTDG Advantage (CTDG AUC - DTDG AUC)",
            title="Does Multigraph Structure Favor CTDG Methods?",
        )

        # Compute correlation
        corr = np.corrcoef(
            merged["compression_ratio"].to_numpy(),
            merged["ctdg_advantage"].to_numpy(),
        )[0, 1]

        correlation_results = {
            "compression_vs_ctdg_advantage": {
                "correlation": float(corr),
                "data": merged.to_dicts(),
            },
        }
    else:
        correlation_results = {"error": "Insufficient data for correlation"}

    # Save correlation results
    with open(os.path.join(output_dir, "correlation_analysis.json"), "w") as f:
        json.dump(correlation_results, f, indent=4, default=float)

    logger.info("Correlation analysis complete")
    return correlation_results


@app.command()
def main(
    n_workers: int = typer.Option(8, help="Number of parallel workers"),
    datasets: list[str] = typer.Option(None, help="Specific datasets to analyze"),
    output_base: str = typer.Option(
        "analysis-results/multigraph", help="Base output directory"
    ),
    experiment_dir: str = typer.Option(
        "experiment-results", help="Directory with benchmark results"
    ),
    skip_correlation: bool = typer.Option(
        False, help="Skip performance correlation analysis"
    ),
) -> None:
    """Run multigraph analysis on all datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    target_datasets = datasets if datasets else DATASETS
    logging.info(f"Analyzing {len(target_datasets)} datasets with {n_workers} workers")

    # Prepare tasks
    tasks = [(ds, os.path.join(output_base, ds)) for ds in target_datasets]

    # Run multiplicity analysis in parallel
    all_results = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for result in executor.map(analyze_multiplicity_wrapper, tasks):
            all_results[result["dataset"]] = result

    # Create summary table
    summary_records = []
    for ds, result in all_results.items():
        if "error" not in result:
            stats = result.get("full_graph", {})
            summary_records.append(
                {
                    "dataset": ds,
                    "total_edges": stats.get("total_edges"),
                    "unique_edges": stats.get("unique_edges"),
                    "compression_ratio": stats.get("compression_ratio"),
                    "pct_repeated": stats.get("pct_repeated"),
                    "max_multiplicity": stats.get("max_multiplicity"),
                    "mean_multiplicity": stats.get("mean_multiplicity"),
                }
            )

    if summary_records:
        summary_df = pl.DataFrame(summary_records)
        summary_df.write_csv(os.path.join(output_base, "multiplicity_summary.csv"))

    # Run correlation analysis
    if not skip_correlation:
        correlation_dir = os.path.join(output_base, "correlation")
        correlate_with_performance(all_results, experiment_dir, correlation_dir)

    # Save overall summary
    summary = {
        "datasets_analyzed": list(all_results.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": {
            ds: r.get("full_graph", {})
            for ds, r in all_results.items()
            if "error" not in r
        },
    }
    with open(os.path.join(output_base, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Analysis complete. Results saved to {output_base}")


if __name__ == "__main__":
    app()
