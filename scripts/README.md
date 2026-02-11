# Scripts

This directory contains two categories of scripts:

1. **Dataset preparation scripts** — one-off scripts for downloading and preprocessing specific datasets (e.g., `bitcoin.py`, `amazon.py`).
2. **Analysis scripts** — post-experiment analysis pipelines that consume benchmark results and produce plots/statistics.

This README focuses on the analysis scripts.

## Prerequisites

```bash
pixi install && pixi shell
export BASE_PATH=$(pwd)
```

All analysis scripts use [Typer](https://typer.tiangolo.com/) for CLI and support `--help`.

## Analysis Pipeline Overview

The analysis experiments form a coherent sequence:

```
Experiment 4a (spectral baseline)
    |
    v
Experiment 4b (spectral shift)  -->  Experiment 7 (spectral-performance correlation)
    |                                       ^
    v                                       |
Experiment 4c (spectral signature)     Aim repo (benchmark results)
    |
    v
Experiment 5 (multigraph analysis)  -->  experiment-results/ (benchmark results)
```

**Experiments 4a-5** can be run together via `run_all_analysis.py`.
**Experiment 7** must be run separately after both the spectral shift analysis (4b) and benchmark experiments have completed.

## Quick Start

```bash
# Run all spectral + multigraph analysis (experiments 4a-5)
python scripts/run_all_analysis.py

# Run spectral-performance correlation (experiment 7)
# Requires: experiment 4b results + Aim-tracked benchmark runs
python scripts/run_spectral_performance.py
```

---

## Script Reference

### `run_all_analysis.py` — Master Orchestrator (Experiments 4a-5)

Runs experiments 4a, 4b, 4c, and 5 in sequence. Each sub-experiment can be skipped individually.

```bash
python scripts/run_all_analysis.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-workers` | `16` | Number of parallel workers |
| `--output-base` | `analysis-results` | Base output directory |
| `--experiment-dir` | `experiment-results` | Directory with benchmark results (for Experiment 5) |
| `--skip-baseline` | `False` | Skip spectral baseline (4a) |
| `--skip-shift` | `False` | Skip spectral shift (4b) |
| `--skip-signature` | `False` | Skip spectral signature (4c) |
| `--skip-multigraph` | `False` | Skip multigraph analysis (5) |
| `--datasets` | all 6 | Specific datasets to analyze (repeatable) |

**Default datasets:** bitcoin-alpha, bitcoin-otc, as-topology, digg-homo, email-dnc, uc-social

---

### `run_spectral_baseline.py` — Experiment 4a: Spectral Baseline

Computes spectral properties (eigenvalues, S_high, spectral density) for the clean (no anomaly) version of each dataset.

```bash
python scripts/run_spectral_baseline.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-workers` | `8` | Number of parallel workers |
| `--datasets` | all 6 | Specific datasets (repeatable) |
| `--output-base` | `analysis-results/spectral` | Output directory |

**Outputs** (per dataset in `{output-base}/{dataset}/baseline/`):
- `eigenvalues.npy` — raw eigenvalue array
- `spectral_metrics.json` — S_high, eigenvalue stats, graph info
- `density.png` / `density.svg` — spectral density histogram

**Cross-dataset outputs** (`{output-base}/plots/`):
- `eigenvalue_violin.png` — violin plot comparing all datasets

**Summary:** `{output-base}/baseline_summary.json`

---

### `run_spectral_shift.py` — Experiment 4b: Spectral Shift

Computes delta S_high (spectral shift) between clean and anomalous graphs for every (dataset, anomaly_type) pair.

```bash
python scripts/run_spectral_shift.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-workers` | `8` | Number of parallel workers |
| `--datasets` | all 6 | Specific datasets (repeatable) |
| `--anom-types` | all 5 | Specific anomaly types (repeatable) |
| `--anom-ratio` | `0.1` | Anomaly injection ratio |
| `--duration` | `medium` | Anomaly duration (small/medium/large) |
| `--output-base` | `analysis-results/spectral` | Output directory |

**Default anomaly types:** random, burst, clique, path, bridge

**Outputs** (per pair in `{output-base}/{dataset}/anomalous/{anom_type}/`):
- `spectral_shift.json` — clean/anom S_high, delta_s_high, graph sizes
- `energy_curves.npz` — clean and anomalous energy ratio curves

**Cross-dataset outputs** (`{output-base}/plots/`):
- `delta_s_high_heatmap.png` — datasets x anomaly types heatmap
- `delta_s_high_bar.png` — grouped bar chart by anomaly type
- `energy_ratio_{dataset}.png` — energy ratio curves per dataset

**Summary:** `{output-base}/shift_summary.json` (contains the `delta_s_high` dict consumed by Experiment 7)

---

### `run_spectral_signature.py` — Experiment 4c: Spectral Signature

Computes full eigenvalue distributions for clean and each anomaly type, analyzing which frequency bands are most affected.

```bash
python scripts/run_spectral_signature.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-workers` | `8` | Number of parallel workers |
| `--datasets` | all 6 | Specific datasets (repeatable) |
| `--anom-types` | all 5 | Specific anomaly types (repeatable) |
| `--anom-ratio` | `0.1` | Anomaly injection ratio |
| `--duration` | `medium` | Anomaly duration (small/medium/large) |
| `--output-base` | `analysis-results/spectral` | Output directory |

**Outputs** (per dataset in `{output-base}/{dataset}/signature/`):
- Per anomaly type: eigenvalues, spectral metrics, histogram comparison plots
- `kde_overlay.png` — KDE overlay of all conditions
- `frequency_bands.png` — frequency band distribution bars
- `signature_summary.json` — per-condition spectral stats

**Cross-dataset outputs** (`{output-base}/plots/`):
- `delta_mean_heatmap.png` — mean eigenvalue shift heatmap
- `concentration_at_one.png` — clique concentration analysis

---

### `run_multigraph_analysis.py` — Experiment 5: Multigraph Handling

Computes edge multiplicity statistics for each dataset and correlates with DTDG vs CTDG method performance.

```bash
python scripts/run_multigraph_analysis.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-workers` | `8` | Number of parallel workers |
| `--datasets` | all 6 | Specific datasets (repeatable) |
| `--output-base` | `analysis-results/multigraph` | Output directory |
| `--experiment-dir` | `experiment-results` | Directory with benchmark results |
| `--skip-correlation` | `False` | Skip performance correlation |

**Method categorization:**
- DTDG (binary deduplication): strgnn, taddy, gcn, gat, graphsage
- CTDG (preserves parallel edges): sad, slade

**Outputs** (per dataset in `{output-base}/{dataset}/`):
- `multiplicity_stats.json` — full and train-split multiplicity statistics
- `distribution.png` — multiplicity distribution bar chart

**Cross-dataset outputs:**
- `multiplicity_summary.csv` — summary table
- `correlation/plots/multiplicity_vs_performance.png` — DTDG vs CTDG scatter
- `correlation/plots/compression_vs_ctdg_advantage.png` — compression ratio correlation
- `correlation/correlation_analysis.json` — Pearson r and data

---

### `run_spectral_performance.py` — Experiment 7: Spectral-Performance Correlation

Correlates spectral shift (delta S_high from Experiment 4b) with per-method anomaly detection ROC-AUC queried from the Aim experiment tracker.

Tests whether spectral-aware methods (TADDY, RustGraph) perform better on datasets/anomalies with larger spectral shifts.

```bash
python scripts/run_spectral_performance.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--aim-repo` | `.` | Path to directory containing the `.aim` repo |
| `--experiment-name` | `None` | Filter to a specific Aim experiment |
| `--spectral-dir` | `analysis-results/spectral` | Directory with Experiment 4b results |
| `--output-base` | `analysis-results/spectral/performance` | Output directory |
| `--anom-ratio` | `0.1` | Must match Experiment 4b's anom_ratio |
| `--anom-duration` | `medium` | Must match Experiment 4b's duration |

**Data sources:**
1. `{spectral-dir}/shift_summary.json` — delta S_high values (from `run_spectral_shift.py`)
2. `.aim/` repository — per-run ROC-AUC and metadata (from `runner.py` benchmark runs)

**Method categorization:**
- Spectral-aware: taddy, rustgraph
- Non-spectral: sad, slade, strgnn, gcn, gat, graphsage, generaldyg, addgraph

**Outputs** (`{output-base}/plots/`):
- `spectral_performance_scatter.png` — main scatter with regression lines for spectral vs non-spectral groups
- `method_correlation.png` — horizontal bar chart of per-method Pearson r with significance markers
- `spectral_performance_grid.png` — small-multiples grid, one scatter per method
- `overall_correlation.png` — pooled scatter across all methods

**Summary:** `{output-base}/correlation_summary.json` — all correlation coefficients, p-values, and raw data

---

## Shell Wrappers

### `run_analysis_internal.sh`

Internal entry point for running the full analysis pipeline. Detects whether it is inside an Apptainer container or a pixi environment and configures accordingly.

```bash
./scripts/run_analysis_internal.sh [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-workers` | `8` | Number of parallel workers |
| `--skip-download` | off | Skip dataset download verification |
| `--skip-baseline` | off | Skip spectral baseline (4a) |
| `--skip-shift` | off | Skip spectral shift (4b) |
| `--skip-signature` | off | Skip spectral signature (4c) |
| `--skip-multigraph` | off | Skip multigraph analysis (5) |

### `run_analysis_server.sh`

Convenience wrapper that runs `run_analysis_internal.sh` through pixi (for non-container environments). Requires pixi and uv.

```bash
./scripts/run_analysis_server.sh
```

---

## Output Directory Structure

After a full analysis run, the output looks like:

```
analysis-results/
  spectral/
    baseline_summary.json
    shift_summary.json           <-- consumed by Experiment 7
    plots/
      eigenvalue_violin.png
      delta_s_high_heatmap.png
      delta_s_high_bar.png
      delta_mean_heatmap.png
      concentration_at_one.png
      energy_ratio_{dataset}.png
    {dataset}/
      baseline/
        eigenvalues.npy
        spectral_metrics.json
        density.png
      anomalous/{anom_type}/
        spectral_shift.json
        energy_curves.npz
      signature/
        ...
    performance/                 <-- Experiment 7
      correlation_summary.json
      plots/
        spectral_performance_scatter.png
        method_correlation.png
        spectral_performance_grid.png
        overall_correlation.png
  multigraph/
    analysis_summary.json
    multiplicity_summary.csv
    {dataset}/
      multiplicity_stats.json
      distribution.png
    correlation/
      correlation_analysis.json
      plots/
        multiplicity_vs_performance.png
        compression_vs_ctdg_advantage.png
```
