# DGADB

Dynamic Graph Anomaly Detection Benchmark — a research framework for evaluating anomaly detection algorithms on temporal/dynamic graphs.

## Setup

[pixi](https://pixi.sh) is required (manages conda + pip dependencies and avoids version conflicts).

```bash
git clone <repo-url> && cd dgadb
pixi install
pixi shell
```

**Apptainer**: On older Linux servers where installing pixi is impractical or when the requirements cannot
be met (old libc or gcc), an [Apptainer](https://apptainer.org/) image is available. Build it with `apptainer
build container/dgadb.sif container/dgadb.def` and run experiments via `./container/run-dgadb.sh`. See
[`container/README.md`](container/README.md) for full details.

## Download Data

Download all datasets at once:

```bash
pixi run download-data
```

Or download individual datasets with [uv](https://docs.astral.sh/uv/) (bundled with pixi):

```bash
uv run scripts/bitcoin.py      # downloads bitcoin-alpha & bitcoin-otc
uv run scripts/uc-social.py
uv run scripts/email-dnc.py
# ... etc.
```

Data is placed under `data/<dataset-name>/` in the project root.

## Run Experiments

```bash
pixi run experiment --method sad --dataset bitcoin-alpha
```

Available methods: `sad`, `taddy`, `slade`, `strgnn`, `rustgraph`, `gcn`, `gat`, `graphsage`, `generaldyg`, `addgraph`

Key options:

| Option | Description |
|--------|-------------|
| `--experiment-name` | Aim experiment name (default: `dgadb`) |
| `--anom-types` | Anomaly types to inject |
| `--anom-rates` | Anomaly rates |
| `--anom-durations` | Anomaly durations |
| `--epochs` | Number of training epochs |
| `--concurrency` | Parallel runs |

Datasets are defined in `configs/datasets/`. Method names must be lowercase.

## Hyperparameter Tuning

Use `tune` (not `experiment`) for hyperparameter tuning via Ray Tune:

```bash
pixi run tune --method GCN GAT --dataset bitcoin-alpha uc-social \
    --config_path configs/experiments/tune.yaml
```

## View Results

Experiments are tracked with [Aim](https://aimstack.io/):

```bash
pixi run aim  # launches UI at http://localhost:43800
```

Logged metrics include per-step loss, per-epoch validation ROC-AUC/AP, and final test metrics.

## Scalability Profiling

`scripts/benchmark_scalability.py` measures setup, training, and inference
cost for each (method, dataset) pair. The inference loop is instrumented
with a `StreamingProfiler` (`src/dgadb/scalability/streaming_profiler.py`)
that records per-snapshot wall-clock latency, excludes a warmup window,
and writes streaming metrics into the result CSV plus a per-run JSON
sidecar.

### Run a single configuration

```bash
pixi run -e dev python -m scripts.benchmark_scalability \
    --methods graphsage \
    --datasets bitcoin-alpha \
    --epochs 1 \
    --device gpu \
    --output-dir benchmark-results/smoke
```

Outputs in `--output-dir`:

- `benchmark_results_<timestamp>.csv` — one row per (method, dataset) with
  legacy columns (`inference_edges_per_sec`, `peak_ram_mb`, ...) plus
  streaming columns (`throughput_warmup_excluded`, `latency_p50_ms`,
  `latency_p95_ms`, `latency_p99_ms`, `latency_p99_over_p50`,
  `snapshots_after_warmup`, `insufficient_batches`).
- `benchmark_results_<timestamp>.json` — the same rows as JSON.
- `sidecars_<timestamp>/<method>__<dataset>.json` — per-run payload with
  `per_snapshot_latency_ms`, `per_snapshot_edges`,
  `per_snapshot_mean_degree`, and `windowed_throughput` (1 s windows).

`insufficient_batches=True` means the test split produced fewer
post-warmup snapshots than the profiler's floor (default
`min_post_warmup=20`). Treat throughput/latency numbers in that row as
unreliable — the dataset is too small for streaming measurements.

### Run the full Tier A sweep

To reproduce the appendix table for the rebuttal, use the dispatch
script:

```bash
./scripts/run_scalability_sweep.sh
```

This runs every (method, dataset) pair one at a time, skipping cells that
§4.5 of the paper already established as `OOM (compute-bound)` or
`OOM (memory-bound)`. Results accumulate in `benchmark-results/tier_a/`
with a per-run log under `benchmark-results/tier_a/logs/`. The script is
idempotent: already-completed cells are detected by the presence of their
log file and skipped on re-runs.

### Generate the appendix artifacts

After the sweep finishes, use `scripts/analyze_streaming_scalability.py`
to produce the table and figures:

```bash
pixi run -e dev python -m scripts.analyze_streaming_scalability tier-a-table \
    --csv-glob 'benchmark-results/tier_a/benchmark_results_*.csv' \
    --output benchmark-results/tier_a/tier_a_table.tex

pixi run -e dev python -m scripts.analyze_streaming_scalability latency-cdf \
    --sidecar-glob 'benchmark-results/tier_a/sidecars_*/*.json' \
    --output benchmark-results/tier_a/latency_cdf.pdf

pixi run -e dev python -m scripts.analyze_streaming_scalability cost-curve \
    --sidecar-glob 'benchmark-results/tier_a/sidecars_*/*.json' \
    --output benchmark-results/tier_a/cost_curve.pdf
```

The `KNOWN_OOM` map inside `analyze_streaming_scalability.py` is the
single source of truth for which cells are inherited from §4.5; extend
it if the sweep reveals additional ceilings.

## Code Quality

```bash
pixi run -e dev lint        # ruff check .
pixi run -e dev format      # ruff format .
pixi run -e dev typecheck   # pyrefly check .
pixi run -e dev check       # lint + typecheck
```

Pre-commit hooks enforce these checks automatically.

## Project Structure

```
configs/             Dataset and experiment YAML configs
data/                Downloaded datasets (git-ignored)
scripts/             Download scripts & analysis pipelines
src/dgadb/
  data/              Dataset loading (load_df)
  evaluation/        ROC-AUC / AP evaluator
  experiment/        ExperimentRunner, tune, callbacks
  models/            AD model implementations
    base.py          BaseADModel interface
    *_new/           Current model implementations
  preprocessing/     Data pipeline
  storage/           TemporalGraph, snapshots, loaders
  utils/             Config loading, paths
```
