# DyGADBench — Experimental Results

This folder contains the complete per-dataset results from the DyGADBench evaluation, complementing the aggregated metrics reported in the main paper (Table 3). The contents are organized into two subdirectories:

```
supplementary-materials/
├── tables/       # Per-method CSV files with full numerical results
├── plots/        # Sensitivity analysis plots (ROC-AUC and AUPRC)
├── browse.py     # Interactive marimo notebook (see below)
└── README.md
```

## Interactive browser

`browse.py` is a [marimo](https://marimo.io) notebook that lets you scan
through the tables and plots without leaving the terminal. Launch it with:

```bash
pixi run marimo edit supplementary-materials/browse.py
# or, for a read-only app view:
pixi run marimo run supplementary-materials/browse.py
```

The notebook has five sections: a multi-axis table filter, an aggregated
summary, a head-to-head leaderboard at a fixed (dataset, anomaly type, R, T)
configuration, a per-method heatmap over the (dataset × R × T) grid, and
an embedded viewer for the sensitivity PDFs.

## Tables

Each CSV file in `tables/` is named `<method>.csv` and contains the following columns:

| Column | Description |
|---|---|
| `method` | Model name |
| `dataset` | One of the 6 evaluation datasets (see below) |
| `anom_type` | Injected anomaly type: `random`, `bridge`, `path`, `burst`, or `clique` |
| `anom_ratio` | Injection rate *R* ∈ {0.01, 0.05, 0.10} |
| `anom_duration` | Temporal span *T* ∈ {0.1, 0.5, 1.0}, as a fraction of the test observation horizon |
| `rocauc` | Area Under the ROC Curve |
| `auprc` | Area Under the Precision–Recall Curve |

Each row corresponds to a single (dataset, anomaly type, *R*, *T*) configuration averaged over three independent runs.

## Plots

The `plots/` directory contains sensitivity analysis figures corresponding to Section 4.3 of the paper. For each anomaly type, we provide plots showing how detection performance varies with the injection rate *R* and the temporal span *T*, reported for both ROC-AUC and AUPRC:

- **Varying *R*** (injection rate): performance across *R* ∈ {0.01, 0.05, 0.10}, aggregated over all datasets and temporal spans, for each anomaly type (random, bridge, path, burst, clique).
- **Varying *T*** (temporal span): performance across *T* ∈ {0.1, 0.5, 1.0}, aggregated over all datasets, for each anomaly type.

The ROC-AUC plots correspond to Figures 1b–1k in the paper. The AUPRC plots provide the same analysis under the Precision–Recall metric.

## Methods

| File | Method | Reference |
|---|---|---|
| `addgraph.csv` | AddGraph | Zheng et al., IJCAI 2019 |
| `rustgraph.csv` | RustGraph | Guo et al., TKDE 2024 |
| `sad.csv` | SAD | Tian et al., IJCAI 2023 |
| `slade.csv` | SLADE | Lee et al., KDD 2024 |
| `strgnn.csv` | StrGNN | Cai et al., CIKM 2021 |
| `taddy.csv` | TADDY | Liu et al., TKDE 2023 |
| `generaldyg.csv` | GeneralDyG | Yang et al., AAAI 2025 |
| `gcn.csv` | GCN (static baseline) | Kipf & Welling, 2016 |
| `gat.csv` | GAT (static baseline) | Veličković et al., 2017 |
| `graphsage.csv` | GraphSAGE (static baseline) | Hamilton et al., NeurIPS 2017 |

## Datasets

The evaluation covers 6 real-world dynamic graph datasets spanning multiple domains:

| Dataset | Domain | Nodes | Edges |
|---|---|---|---|
| Bitcoin-Alpha | Trade | 3,782 | 24,136 |
| Bitcoin-OTC | Trade | 5,881 | 35,592 |
| Email-DNC | Interaction | 1,866 | 39,264 |
| UCI Messages | Social | 1,899 | 59,835 |
| Digg | Social | 30,360 | 85,155 |
| AS-Topology | Internet | 34,761 | 171,420 |
