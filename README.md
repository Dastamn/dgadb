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

```bash
python -m scripts.benchmark_scalability \
    --methods sad slade rustgraph \
    --datasets bitcoin-alpha bitcoin-otc wiki mooc reddit \
    --epochs 3
```

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
