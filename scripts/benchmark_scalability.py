"""
Scalability benchmark for DGADB anomaly detection methods.

Measures wall-clock training time, peak GPU/RAM memory, and throughput
(edges/sec) across datasets of increasing size, for each method.

Usage:
    # Single method, multiple datasets (ascending size order)
    python -m scripts.benchmark_scalability \
        --methods sad slade rustgraph \
        --datasets bitcoin-alpha bitcoin-otc wiki mooc reddit \
        --epochs 3

    # Quick smoke test
    python -m scripts.benchmark_scalability \
        --methods gcn --datasets bitcoin-alpha --epochs 1
"""

import gc
import json
import resource
from pathlib import Path
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Annotated, List, Optional
import datetime

import torch
import typer
import polars as pl
from loguru import logger

from dgadb.experiment.runner import Method
from dgadb.storage.temporal_graph import TemporalGraph, TemporalGraphLoaderNew
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshotLoader
from dgadb.experiment.callbacks import ExperimentCallback
from dgadb.models.base import TrainingState

# Configure loguru: remove default handler, add one with a clean format
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}")


def _configure_torch_threads() -> None:
    """Honor ``DGADB_NUM_THREADS`` / ``OMP_NUM_THREADS`` for PyTorch CPU ops.

    PyTorch's default CPU thread count depends on how the wheel was built and
    on ``OMP_NUM_THREADS`` at import time. On some pixi environments that is
    1, which makes full-graph GCN/GAT/GraphSAGE forwards on epinions
    pathologically slow. We resolve the intended count from environment
    variables (``DGADB_NUM_THREADS`` wins, then ``OMP_NUM_THREADS``, then
    ``MKL_NUM_THREADS``, finally ``os.cpu_count()``) and force it explicitly.
    """
    requested = (
        os.environ.get("DGADB_NUM_THREADS")
        or os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("MKL_NUM_THREADS")
    )
    try:
        n = int(requested) if requested else (os.cpu_count() or 1)
    except ValueError:
        n = os.cpu_count() or 1
    n = max(1, n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(min(n, 4))
    except RuntimeError:
        # set_num_interop_threads must be called before any parallel work.
        # If something already spawned the pool, skip silently.
        pass
    logger.info(f"Torch CPU threads: num_threads={torch.get_num_threads()}, "
                f"num_interop_threads={torch.get_num_interop_threads()}")


_configure_torch_threads()


def _peak_rss_mb() -> float:
    """Return the process peak resident set size in MiB.

    Uses ``resource.getrusage(RUSAGE_SELF).ru_maxrss``. The unit reported by
    that syscall is KiB on Linux and bytes on macOS, so we normalize based on
    platform.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxrss / (1024 ** 2)  # bytes -> MiB
    return maxrss / 1024  # KiB -> MiB

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
    
    # New metrics
    inference_time_sec: float = 0.0
    inference_edges_per_sec: float = 0.0

    # Streaming metrics (warmup-excluded)
    throughput_warmup_excluded: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_p99_over_p50: float = 0.0
    snapshots_after_warmup: int = 0
    insufficient_batches: bool = False

    # Detection delay (buffering + scoring), in seconds. Reported only for
    # methods whose inference loop iterates over snapshots in the natural
    # way and exposes per-snapshot data-time spans (i.e. base.py
    # run_inference; methods that override the loop with a batched form do
    # not report these and the values stay at 0.0).
    detection_delay_mean_sec: float = 0.0
    detection_delay_max_sec: float = 0.0
    detection_delay_p95_sec: float = 0.0
    detection_delay_p99_sec: float = 0.0
    snapshot_span_mean_sec: float = 0.0
    snapshot_span_max_sec: float = 0.0
    snapshots_with_span: int = 0

    time_to_convergence_sec: float = 0.0
    best_val_auc: float = 0.0
    best_epoch: int = -1
    
    error: str | None = None


class ConvergenceMonitor(ExperimentCallback):
    """Tracks training progress to find time-to-convergence."""
    def __init__(self):
        super().__init__()
        self.start_time = 0.0
        self.best_val_auc = -1.0
        self.best_epoch = -1
        self.time_to_best = 0.0
        self.history = []

    def on_train_begin(self, state: TrainingState):
        self.start_time = time.perf_counter()

    def on_train_epoch_end(self, state: TrainingState):
        elapsed = time.perf_counter() - self.start_time
        val_auc = state.val_metrics.get('roc_auc', 0.0)
        
        self.history.append({
            'epoch': state.epoch,
            'time': elapsed,
            'val_auc': val_auc
        })

        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = state.epoch
            self.time_to_best = elapsed


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
    """Run a single benchmark: setup + train + inference, recording time and memory."""
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

    # Baseline peak RSS before the benchmark allocates anything. ``ru_maxrss``
    # is monotonically non-decreasing per process, so taking the delta against
    # a baseline gives a conservative estimate of this benchmark's peak
    # allocation even when a prior benchmark in the same process already
    # pushed the peak up.
    rss_baseline_mb = _peak_rss_mb()
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
        
        monitor = ConvergenceMonitor()

        t_train_start = time.perf_counter()
        model.train(epochs, train_loader, val_loader, callbacks=[monitor])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_train_end = time.perf_counter()

        result.train_time_sec = t_train_end - t_train_start
        result.total_time_sec = result.setup_time_sec + result.train_time_sec
        result.time_to_convergence_sec = monitor.time_to_best
        result.best_val_auc = monitor.best_val_auc
        result.best_epoch = monitor.best_epoch

        # Throughput: edges processed during training
        train_edges = int(data.train_mask.sum().item())
        total_edges_processed = train_edges * epochs
        result.edges_per_sec = total_edges_processed / max(result.train_time_sec, 1e-9)

        # --- Inference Phase ---
        logger.info("    Running inference...")
        from dgadb.scalability.streaming_profiler import StreamingProfiler
        profiler = StreamingProfiler()

        test_loader = TemporalGraphSnapshotLoader(data, split="test", **snap_config)
        t_inf_start = time.perf_counter()
        model.run_inference(test_loader, profiler=profiler)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_inf_end = time.perf_counter()

        result.inference_time_sec = t_inf_end - t_inf_start

        # Legacy whole-test-set ratio, kept for backwards compatibility.
        test_edges = int(data.test_mask.sum().item())
        result.inference_edges_per_sec = test_edges / max(result.inference_time_sec, 1e-9)

        # Streaming metrics (warmup-excluded).
        summary = profiler.summary()
        result.throughput_warmup_excluded = summary["throughput_warmup_excluded"]
        result.latency_mean_ms = summary["latency_mean_ms"]
        result.latency_p50_ms = summary["latency_p50_ms"]
        result.latency_p95_ms = summary["latency_p95_ms"]
        result.latency_p99_ms = summary["latency_p99_ms"]
        result.latency_p99_over_p50 = summary["latency_p99_over_p50"]
        result.snapshots_after_warmup = summary["snapshots_after_warmup"]
        result.insufficient_batches = summary["insufficient_batches"]

        # Detection-delay metrics. Present only when at least one snapshot's
        # data-time span was supplied to the profiler (i.e. methods that go
        # through base.py run_inference); methods that override the loop
        # with a batched form leave the keys absent and we keep the result
        # defaults.
        if "snapshots_with_span" in summary:
            result.snapshots_with_span = summary["snapshots_with_span"]
            result.detection_delay_mean_sec = summary["detection_delay_mean_sec"]
            result.detection_delay_max_sec = summary["detection_delay_max_sec"]
            result.detection_delay_p95_sec = summary["detection_delay_p95_sec"]
            result.detection_delay_p99_sec = summary["detection_delay_p99_sec"]
            result.snapshot_span_mean_sec = summary["snapshot_span_mean_sec"]
            result.snapshot_span_max_sec = summary["snapshot_span_max_sec"]

        # Stash sidecar on the result for the writer phase.
        result._sidecar = profiler.to_sidecar()  # type: ignore[attr-defined]

    except Exception as e:
        logger.opt(exception=True).error(f"Error benchmarking {method.value} on {dataset_name}: {e}")
        result.error = str(e)

    # Memory measurements. ``ru_maxrss`` from ``getrusage`` reports true peak
    # RSS, unlike ``process.memory_info().rss`` which reports instantaneous
    # RSS and therefore misses peaks that have already been freed by the time
    # we sample. Subtract the baseline so we get *this* benchmark's additional
    # footprint, even when prior benchmarks in the same process pushed the
    # peak up.
    result.peak_ram_mb = max(0.0, _peak_rss_mb() - rss_baseline_mb)
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
    # Hard-fail when the caller asked for GPU but CUDA is not actually
    # available. The previous silent fallback to CPU cost us hours of wall
    # time on a sweep that was supposed to run on GPU but was using the
    # `dev` pixi env (no CUDA-enabled torch) instead of `cuda`; we'd rather
    # exit loudly here than produce CPU numbers masquerading as GPU numbers.
    if device_str == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device gpu was requested but torch.cuda.is_available() is False. "
            "Check that PyTorch was installed with CUDA support (e.g. pixi env "
            "`cuda` rather than `dev`) and that the host has a visible GPU "
            "(try `nvidia-smi`)."
        )
    device = torch.device("cuda" if device_str == "gpu" else "cpu")
    logger.info(f"Benchmarking on device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        logger.info(f"CUDA version: {torch.version.cuda}  torch: {torch.__version__}")

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
                f"inf={result.inference_time_sec:.1f}s "
                f"edges/s={result.edges_per_sec:.0f} "
                f"RAM={result.peak_ram_mb:.0f}MB GPU={result.peak_gpu_mb:.0f}MB"
            )

    out_dir = Path(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run sidecars (windowed throughput series, per-snapshot arrays).
    sidecar_dir = out_dir / f"sidecars_{timestamp}"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        sidecar = getattr(r, "_sidecar", None)
        if sidecar is None:
            continue
        slug = f"{r.method}__{r.dataset}".replace("/", "_")
        (sidecar_dir / f"{slug}.json").write_text(json.dumps(sidecar))
    logger.info(f"Sidecars saved to {sidecar_dir}")

    # Strip the private sidecar attribute so asdict() doesn't see it.
    for r in results:
        if hasattr(r, "_sidecar"):
            delattr(r, "_sidecar")

    # Save results
    results_dicts = [asdict(r) for r in results]

    json_path = out_dir / f"benchmark_results_{timestamp}.json"
    json_path.write_text(json.dumps(results_dicts, indent=2, default=str))
    logger.info(f"JSON results saved to {json_path}")

    df = pl.DataFrame(results_dicts)
    csv_path = out_dir / f"benchmark_results_{timestamp}.csv"
    df.write_csv(csv_path)
    logger.info(f"CSV results saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 140)  # Extended line length
    print("SCALABILITY BENCHMARK RESULTS")
    print("=" * 140)

    summary_cols = ["method", "dataset", "num_nodes", "num_edges",
                    "setup_time_sec", "train_time_sec", "inference_time_sec", 
                    "time_to_convergence_sec", "best_val_auc",
                    "edges_per_sec", "peak_ram_mb", "peak_gpu_mb", "error"]
    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df.select(available_cols)

    # Format numeric columns
    for col in ["setup_time_sec", "train_time_sec", "inference_time_sec", "time_to_convergence_sec", "peak_ram_mb", "peak_gpu_mb"]:
        if col in summary.columns:
            summary = summary.with_columns(pl.col(col).round(1))
    
    if "best_val_auc" in summary.columns:
        summary = summary.with_columns(pl.col("best_val_auc").round(4))

    if "edges_per_sec" in summary.columns:
        summary = summary.with_columns(pl.col("edges_per_sec").round(0))

    print(summary)
    print("=" * 140)


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