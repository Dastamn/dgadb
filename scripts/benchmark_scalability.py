"""
Scalability benchmark for DGADB anomaly detection methods.

Measures wall-clock training time, peak GPU/RAM memory, and throughput
(edges/sec) across datasets of increasing size, for each method.

Usage:
    # Single method, multiple datasets (ascending size order)
    BASE_PATH=$PWD python -m scripts.benchmark_scalability \
        --methods sad slade rustgraph \
        --datasets bitcoin-alpha bitcoin-otc wiki mooc reddit \
        --epochs 3

    # Quick smoke test
    BASE_PATH=$PWD python -m scripts.benchmark_scalability \
        --methods gcn --datasets bitcoin-alpha --epochs 1
"""

import gc
import json
from pathlib import Path
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Annotated, List
import datetime

import torch
import typer
import psutil
import polars as pl
from loguru import logger

from dgadb.experiment.runner import Method
from dgadb.storage.temporal_graph import TemporalGraph, TemporalGraphLoaderNew
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshotLoader

# Configure loguru: remove default handler, add one with a clean format
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}")

app = typer.Typer(pretty_exceptions_enable=False)


@dataclass
class BenchmarkResult:
    method: str
    dataset: str
    num_nodes: int
    num_edges: int
    epochs: int
    device: str
    setup_time_sec: float = 0.0
    train_time_sec: float = 0.0
    total_time_sec: float = 0.0
    peak_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    edges_per_sec: float = 0.0
    error: str | None = None


def get_peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_model(method: Method, device: torch.device, window_size: int, cache_dir: str):
    """Instantiate a model for the given method."""
    match method:
        case Method.sad:
            from dgadb.models.sad_new.sad import SADAD
            return SADAD(device=device)
        case Method.taddy:
            from dgadb.models.taddy_new.taddy import TADDYAD
            return TADDYAD(snap_size=window_size, cache_dir=cache_dir, device=device)
        case Method.slade:
            from dgadb.models.slade_new.slade import SLADEAD
            return SLADEAD(device=device)
        case Method.strgnn:
            from dgadb.models.StrGNN.strgnn import StrGNNAD
            return StrGNNAD(snap_size=window_size, cache_dir=cache_dir, device=device)
        case Method.rustgraph:
            from dgadb.models.rustgraph_new.rustgraph import RustGraphAD
            return RustGraphAD(device=device)
        case Method.generaldyg:
            from dgadb.models.generaldyg_new.generaldyg import GeneralDyGAD
            return GeneralDyGAD(cache_dir=cache_dir, device=device)
        case Method.gcn | Method.gat | Method.graphsage:
            from dgadb.models.baseline.gnn import GNNAD
            return GNNAD(method.value.upper(), device=device)
        case Method.addgraph:
            from dgadb.models.addgraph.addgraph import AddGraphAD
            return AddGraphAD(device=device)
        case _:
            raise ValueError(f"Unknown method: {method}")


def load_data(dataset_name: str, anom_type: str = "random",
              anom_ratio: float = 0.05, duration: str = "medium") -> TemporalGraph:
    """Load a dataset with anomaly injection."""
    loader = TemporalGraphLoaderNew()
    return loader.load(
        dataset_name,
        anom_type=anom_type,
        anom_val_ratio=anom_ratio,
        anom_test_ratio=anom_ratio,
        duration=duration,
        create_if_not_found=True,
    )


def benchmark_single(
    method: Method,
    dataset_name: str,
    data: TemporalGraph,
    epochs: int,
    device: torch.device,
    cache_dir: str,
) -> BenchmarkResult:
    """Run a single benchmark: setup + train, recording time and memory."""
    window_size = 2000 if dataset_name in ["bitcoin-alpha", "bitcoin-otc", "uc-social"] else 6000
    include_cumulative = method in [Method.gcn, Method.gat, Method.graphsage]

    result = BenchmarkResult(
        method=method.value,
        dataset=dataset_name,
        num_nodes=data.num_nodes,
        num_edges=data.num_edges,
        epochs=epochs,
        device=str(device),
    )

    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 ** 2)
    reset_gpu_stats()

    snap_config = {
        "strategy": "window",
        "window_size": window_size,
        "include_cumulative": include_cumulative,
    }

    model = None
    try:
        model = get_model(method, device, window_size, cache_dir)

        # --- Setup phase ---
        t_setup_start = time.perf_counter()
        model.setup(data)
        t_setup_end = time.perf_counter()
        result.setup_time_sec = t_setup_end - t_setup_start

        # --- Training phase ---
        train_loader = TemporalGraphSnapshotLoader(data, split="train", **snap_config)
        val_loader = TemporalGraphSnapshotLoader(data, split="val", **snap_config)

        t_train_start = time.perf_counter()
        model.train(epochs, train_loader, val_loader)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_train_end = time.perf_counter()

        result.train_time_sec = t_train_end - t_train_start
        result.total_time_sec = result.setup_time_sec + result.train_time_sec

        # Throughput: edges processed during training
        train_edges = int(data.train_mask.sum().item())
        total_edges_processed = train_edges * epochs
        result.edges_per_sec = total_edges_processed / max(result.train_time_sec, 1e-9)

    except Exception as e:
        logger.opt(exception=True).error(f"Error benchmarking {method.value} on {dataset_name}: {e}")
        result.error = str(e)

    # Memory measurements
    ram_after = process.memory_info().rss / (1024 ** 2)
    result.peak_ram_mb = ram_after - ram_before
    result.peak_gpu_mb = get_peak_gpu_memory_mb()

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_benchmarks(
    methods: list[str],
    datasets: list[str],
    epochs: int,
    device_str: str,
    output_dir: str,
    cache_dir: str,
    anom_type: str,
    anom_ratio: float,
    duration: str,
):
    device = torch.device("cuda" if device_str == "gpu" and torch.cuda.is_available() else "cpu")
    logger.info(f"Benchmarking on device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    os.makedirs(output_dir, exist_ok=True)
    results: list[BenchmarkResult] = []

    for dataset_name in datasets:
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            data = load_data(dataset_name, anom_type, anom_ratio, duration)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to load {dataset_name}: {e}")
            for method_name in methods:
                results.append(BenchmarkResult(
                    method=method_name, dataset=dataset_name,
                    num_nodes=0, num_edges=0, epochs=epochs,
                    device=str(device), error=f"Load failed: {e}",
                ))
            continue

        logger.info(f"  {dataset_name}: {data.num_nodes} nodes, {data.num_edges} edges")

        for method_name in methods:
            method = Method(method_name)
            logger.info(f"  Benchmarking {method.value} on {dataset_name}...")
            result = benchmark_single(method, dataset_name, data, epochs, device, cache_dir)
            results.append(result)

            status = "OK" if result.error is None else f"FAILED: {result.error}"
            logger.info(
                f"    {status} | setup={result.setup_time_sec:.1f}s "
                f"train={result.train_time_sec:.1f}s "
                f"edges/s={result.edges_per_sec:.0f} "
                f"RAM={result.peak_ram_mb:.0f}MB GPU={result.peak_gpu_mb:.0f}MB"
            )

    # Save results
    results_dicts = [asdict(r) for r in results]

    out_dir = Path(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir /  f"benchmark_results_{timestamp}.json"
    json_path.write_text(json.dumps(results_dicts, indent=2, default=str))
    logger.info(f"JSON results saved to {json_path}")

    df = pl.DataFrame(results_dicts)
    csv_path = out_dir / f"benchmark_results_{timestamp}.csv"
    df.write_csv(csv_path)
    logger.info(f"CSV results saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("SCALABILITY BENCHMARK RESULTS")
    print("=" * 100)

    summary_cols = ["method", "dataset", "num_nodes", "num_edges",
                    "setup_time_sec", "train_time_sec", "edges_per_sec",
                    "peak_ram_mb", "peak_gpu_mb", "error"]
    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df.select(available_cols)

    # Format numeric columns
    for col in ["setup_time_sec", "train_time_sec", "peak_ram_mb", "peak_gpu_mb"]:
        if col in summary.columns:
            summary = summary.with_columns(pl.col(col).round(1))
    if "edges_per_sec" in summary.columns:
        summary = summary.with_columns(pl.col("edges_per_sec").round(0))

    print(summary)
    print("=" * 100)


@app.command()
def main(
    methods: Annotated[List[str], typer.Option(help="Methods to benchmark")],
    datasets: Annotated[List[str], typer.Option(help="Datasets to benchmark (in order of expected size)")],
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 3,
    device: Annotated[str, typer.Option(help="Device to use (cpu or gpu)")] = "cpu",
    output_dir: Annotated[str, typer.Option(help="Output directory for results")] = "benchmark-results",
    cache_dir: Annotated[str, typer.Option(help="Cache directory for intermediate files")] = "cache",
    anom_type: Annotated[str, typer.Option(help="Anomaly type for benchmarking")] = "random",
    anom_ratio: Annotated[float, typer.Option(help="Anomaly ratio for val/test splits")] = 0.05,
    duration: Annotated[str, typer.Option(help="Anomaly duration")] = "medium",
):
    # Validate methods
    valid_methods = {m.value for m in Method}
    for m in methods:
        if m not in valid_methods:
            logger.error(f"Unknown method '{m}'. Valid: {sorted(valid_methods)}")
            raise typer.Exit(code=1)

    run_benchmarks(
        methods=methods,
        datasets=datasets,
        epochs=epochs,
        device_str=device,
        output_dir=output_dir,
        cache_dir=cache_dir,
        anom_type=anom_type,
        anom_ratio=anom_ratio,
        duration=duration,
    )


if __name__ == "__main__":
    app()
