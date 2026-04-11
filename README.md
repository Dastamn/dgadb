# DyGADBench — Dynamic Graph Anomaly Detection Benchmark

DyGADBench is a research benchmark for evaluating anomaly detection algorithms on temporal (dynamic) graphs. Given a sequence of graph snapshots, the benchmark injects synthetic anomalies at controlled rates and durations, runs one or more detection methods, and reports edge-level ROC-AUC and average precision. The distinguishing property of DGADB is that anomaly injection, training, and evaluation are fully parameterized through configuration files, so every reported number is reproducible by re-running the same command with the same config.

## Installation

[pixi](https://pixi.sh) is required. It manages the conda and pip dependency stack and avoids version conflicts that arise when installing PyTorch Geometric and its optional dependencies by hand.

```bash
git clone <repo-url> dgadb
cd dgadb
pixi install
pixi shell
```

**On older Linux servers** where installing pixi is impractical (old libc or gcc), an [Apptainer](https://apptainer.org/) image is available. Build it with:

```bash
apptainer build container/dgadb.sif container/dgadb.def
```

and run experiments via `./container/run-dgadb.sh`. See [`container/README.md`](container/README.md) for details.

## Quickstart

Download the Bitcoin-Alpha dataset, then run a five-epoch GCN experiment:

```bash
uv run scripts/bitcoin.py   # downloads bitcoin-alpha and bitcoin-otc
```

```bash
pixi run dgadb run --method gcn --datasets bitcoin-alpha \
  --experiment-name quickstart --anom-types random --anom-rates 0.1 \
  --anom-durations small --epochs 5 --concurrency 1
```

The run completes in under two minutes on a laptop CPU. Results are written to:

```
experiment-results/quickstart/gcn/<variant>/evaluation.json
```

Open `evaluation.json` and look for `"roc_auc"` in the `summary` block. A value above 0.5 confirms the pipeline ran end-to-end correctly. Experiment metrics are also tracked in [Aim](https://aimstack.io/); launch the UI with:

```bash
pixi run aim   # opens http://localhost:43800
```

## Architecture Overview

Data flows through five stages. **Storage** (`src/dgadb/storage/`) defines `TemporalGraph` and `TemporalGraphSnapshot`, the core data structures used throughout. **Preprocessing** (`src/dgadb/preprocessing/`) normalizes raw edge lists and splits them into train/val/test snapshots via `TemporalGraphLoaderNew`; the `_New` suffix marks the current iteration of classes that superseded earlier versions during development. **Anomaly injection** (`src/dgadb/anomaly_injection/`) inserts synthetic anomalous edges into the test split at the rate and duration specified by the config. **Models** (`src/dgadb/models/`) implement `BaseADModel` (`src/dgadb/models/base.py`), which defines the three-method interface — `setup(data)`, `_train_step(snapshot) -> float`, and `_predict(snapshot) -> Tensor`; current model adapters live in `*_new/` subdirectories and are the only ones wired into the runner. **Experiment runner and evaluator** (`src/dgadb/experiment/runner.py`, `src/dgadb/evaluation/evaluator.py`) orchestrate training and compute the final metrics, dispatching events to callbacks (`AimCallback`, `ResourceMonitor`) at each hook point.

## Adding a New Dataset

1. Download the raw edge list and place it under `data/<dataset-name>/`.
2. Write a download script under `scripts/` following the pattern in `scripts/bitcoin.py`.
3. Create a dataset config at `configs/datasets/<dataset-name>.yaml`. Copy `configs/datasets/bitcoin-alpha.yaml` as a starting point; set `dataset.name` and adjust the `pipeline.steps` parameters (split ratios, directionality, etc.) as needed.
4. Verify the dataset loads by running the quickstart command with `--datasets <dataset-name>` and any method.
5. Add the download step to the `download-data` task in `pyproject.toml` if the dataset should be part of the full download sweep.

## Adding a New Method

1. Create a directory `src/dgadb/models/<MethodName>_new/`.
2. In that directory, implement a class that inherits from `BaseADModel` (defined in `src/dgadb/models/base.py`). You must implement `setup(data)`, `_train_step(snapshot)`, `_predict(snapshot)`, `save(save_dir)`, and `load(load_dir)`. See `src/dgadb/models/sad_new/sad.py` for a minimal example.
3. Register the method in the method registry used by the runner. Follow the pattern for an existing method in `src/dgadb/experiment/runner.py`.
4. Smoke-test with: `pixi run dgadb run --method <methodname> --datasets bitcoin-alpha --epochs 1 --concurrency 1`.

## Reproducing the Paper's Experiments

All dataset configs are in `configs/datasets/` and experiment configs in `configs/experiments/`. To run the full evaluation sweep for a given method and dataset set:

```bash
pixi run dgadb run --method sad taddy slade strgnn rustgraph gcn gat graphsage generaldyg addgraph \
  --datasets bitcoin-alpha bitcoin-otc uc-social email-dnc \
  --anom-types random local global \
  --anom-rates 0.05 0.1 0.2 \
  --anom-durations small medium large \
  --epochs 50 --concurrency 4
```

Scalability profiling (Appendix, Tier A table) uses the `scalability` subcommand:

```bash
pixi run dgadb scalability --methods gcn --datasets bitcoin-alpha --epochs 1 --device cpu \
  --output-dir benchmark-results/smoke
```

To reproduce the full Tier A sweep, use the dispatch script:

```bash
./scripts/run_scalability_sweep.sh
```

After the sweep, generate the LaTeX table and figures with `scripts/analyze_streaming_scalability.py` (see the script's `--help` for subcommand options: `tier-a-table`, `latency-cdf`, `cost-curve`).

## Citation and License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
