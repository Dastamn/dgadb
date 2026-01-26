"""Visualization functions for spectral and multigraph analysis.

This module provides plotting functions for visualizing analysis results
from spectral and multigraph experiments.
"""

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Publication-quality color palettes
COLORS = {
    "primary": "#2E86AB",  # Steel blue
    "secondary": "#A23B72",  # Raspberry
    "tertiary": "#F18F01",  # Orange
    "quaternary": "#C73E1D",  # Vermilion
    "quinary": "#3B1F2B",  # Dark purple
    "clean": "#2D3142",  # Gunmetal (for baseline/clean data)
    "grid": "#E8E8E8",  # Light gray for grids
}

# Colorblind-friendly palette for categorical data
CATEGORICAL_COLORS = [
    "#0077BB",  # Blue
    "#33BBEE",  # Cyan
    "#009988",  # Teal
    "#EE7733",  # Orange
    "#CC3311",  # Red
    "#EE3377",  # Magenta
    "#BBBBBB",  # Gray
]


def _setup_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.dpi": 100,
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            # Axes
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            # Ticks
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CCCCCC",
            "legend.fontsize": 10,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


def _save_figure(fig: plt.Figure, output_path: str, dpi: int = 300) -> None:
    """Save figure in both PNG and SVG formats.

    Args:
        fig: Matplotlib figure to save.
        output_path: Base path for output (extension will be replaced).
        dpi: DPI for PNG output.
    """
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)

    # Save PNG (high resolution)
    png_path = path.with_suffix(".png")
    fig.savefig(
        png_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )

    # Save SVG (vector format)
    svg_path = path.with_suffix(".svg")
    fig.savefig(
        svg_path, format="svg", bbox_inches="tight", facecolor="white", edgecolor="none"
    )

    plt.close(fig)


def plot_eigenvalue_violin(
    results: dict[str, np.ndarray],
    output_path: str,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Create violin plot comparing eigenvalue distributions across datasets.

    Args:
        results: Dictionary mapping dataset names to eigenvalue arrays.
        output_path: Path to save the plot.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    datasets = list(results.keys())
    data = [results[ds] for ds in datasets]

    # Create violin plot
    parts = ax.violinplot(
        data,
        positions=range(len(datasets)),
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Style violin bodies
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)])
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.8)
        pc.set_alpha(0.7)

    # Add box plot inside violin for quartiles
    ax.boxplot(
        data,
        positions=range(len(datasets)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": "#333333", "edgecolor": "#333333", "linewidth": 0.8},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
    )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(
        [ds.replace("-", "\n") for ds in datasets], rotation=0, ha="center"
    )
    ax.set_xlabel("Dataset", fontweight="medium")
    ax.set_ylabel("Eigenvalue (Normalized Laplacian)", fontweight="medium")
    ax.set_title("Eigenvalue Distribution Across Datasets")
    ax.set_ylim(-0.05, 2.05)

    # Add horizontal reference lines
    ax.axhline(y=0, color="#AAAAAA", linestyle="-", linewidth=0.5, zorder=0)
    ax.axhline(y=1, color="#AAAAAA", linestyle="--", linewidth=0.5, zorder=0, alpha=0.5)
    ax.axhline(y=2, color="#AAAAAA", linestyle="-", linewidth=0.5, zorder=0)

    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_spectral_density(
    eigenvalues: np.ndarray,
    output_path: str,
    dataset: str,
    bins: int = 40,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Plot histogram of eigenvalue distribution with KDE overlay.

    Args:
        eigenvalues: Array of eigenvalues.
        output_path: Path to save the plot.
        dataset: Dataset name for title.
        bins: Number of histogram bins.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram with cleaner styling
    n, bin_edges, patches = ax.hist(
        eigenvalues,
        bins=bins,
        range=(0, 2),
        density=True,
        alpha=0.7,
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add KDE-like smooth line
    from scipy.ndimage import gaussian_filter1d

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed = gaussian_filter1d(n, sigma=1.5)
    ax.plot(bin_centers, smoothed, color=COLORS["secondary"], linewidth=2, alpha=0.8)

    ax.set_xlabel("Eigenvalue", fontweight="medium")
    ax.set_ylabel("Density", fontweight="medium")
    ax.set_title(f"Spectral Density: {dataset}")
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(bottom=0)

    # Add vertical line at eigenvalue = 1 (center of spectrum)
    ax.axvline(x=1, color="#AAAAAA", linestyle="--", linewidth=1, alpha=0.5)

    # Add statistics annotation
    stats_text = (
        f"n = {len(eigenvalues)}\n"
        f"mean = {eigenvalues.mean():.3f}\n"
        f"std = {eigenvalues.std():.3f}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.9,
        },
    )

    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_delta_s_high_heatmap(
    results: dict[str, dict[str, float]],
    output_path: str,
    figsize: tuple[float, float] = (9, 7),
) -> None:
    """Create heatmap of delta S_high values.

    Args:
        results: Nested dict: {dataset: {anom_type: delta_s_high}}.
        output_path: Path to save the plot.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    datasets = sorted(results.keys())
    anom_types = sorted({at for ds_results in results.values() for at in ds_results})

    # Create matrix
    matrix = np.zeros((len(datasets), len(anom_types)))
    for i, ds in enumerate(datasets):
        for j, at in enumerate(anom_types):
            matrix[i, j] = results.get(ds, {}).get(at, np.nan)

    fig, ax = plt.subplots(figsize=figsize)

    # Use diverging colormap centered at 0
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    # Styling
    ax.set_xticks(range(len(anom_types)))
    ax.set_xticklabels([at.capitalize() for at in anom_types], rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([ds.replace("-", " ").title() for ds in datasets])
    ax.set_xlabel("Anomaly Type", fontweight="medium")
    ax.set_ylabel("Dataset", fontweight="medium")
    ax.set_title("Spectral Shift ($\\Delta S_{high}$) Under Anomaly Injection")

    # Add colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label("$\\Delta S_{high}$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Add value annotations with adaptive text color
    for i in range(len(datasets)):
        for j in range(len(anom_types)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="medium",
                )

    # Add cell borders
    for i in range(len(datasets) + 1):
        ax.axhline(y=i - 0.5, color="white", linewidth=1)
    for j in range(len(anom_types) + 1):
        ax.axvline(x=j - 0.5, color="white", linewidth=1)

    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_delta_s_high_grouped_bar(
    results: dict[str, dict[str, float]],
    output_path: str,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Create grouped bar chart of delta S_high values.

    Args:
        results: Nested dict: {dataset: {anom_type: delta_s_high}}.
        output_path: Path to save the plot.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    datasets = sorted(results.keys())
    anom_types = sorted({at for ds_results in results.values() for at in ds_results})

    x = np.arange(len(datasets))
    width = 0.8 / len(anom_types)

    fig, ax = plt.subplots(figsize=figsize)

    for i, at in enumerate(anom_types):
        values = [results.get(ds, {}).get(at, 0) for ds in datasets]
        offset = (i - len(anom_types) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width * 0.9,
            label=at.capitalize(),
            color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Dataset", fontweight="medium")
    ax.set_ylabel("$\\Delta S_{high}$", fontweight="medium")
    ax.set_title("Spectral Shift by Anomaly Type")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace("-", "\n") for ds in datasets], ha="center")
    ax.legend(title="Anomaly Type", loc="upper right", framealpha=0.95)
    ax.axhline(y=0, color="#333333", linestyle="-", linewidth=0.8)

    # Adjust y-axis to show small values clearly
    ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.set_ylim(-ymax * 1.1, ymax * 1.1)

    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_energy_ratio_curves(
    clean: tuple[np.ndarray, np.ndarray],
    anomalous: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str,
    dataset: str | None = None,
    figsize: tuple[float, float] = (9, 6),
) -> None:
    """Plot cumulative energy ratio curves for clean and anomalous graphs.

    Args:
        clean: Tuple of (eigenvalues, cumulative_energy) for clean graph.
        anomalous: Dict mapping anom_type to (eigenvalues, cumulative_energy).
        output_path: Path to save the plot.
        dataset: Optional dataset name for title.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot clean baseline
    eigenvalues, energy = clean
    ax.plot(
        eigenvalues,
        energy,
        label="Clean (Baseline)",
        linewidth=2.5,
        color=COLORS["clean"],
        zorder=10,
    )

    # Plot anomalous variants
    for i, (anom_type, (eigenvalues, energy)) in enumerate(sorted(anomalous.items())):
        ax.plot(
            eigenvalues,
            energy,
            label=anom_type.capitalize(),
            linewidth=1.8,
            color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
            alpha=0.85,
        )

    ax.set_xlabel("Eigenvalue", fontweight="medium")
    ax.set_ylabel("Cumulative Energy Ratio", fontweight="medium")
    title = "Cumulative Energy Ratio vs Frequency"
    if dataset:
        title = f"{title}\n{dataset.replace('-', ' ').title()}"
    ax.set_title(title)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)

    # Reference lines
    ax.axhline(y=0.5, color="#AAAAAA", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(y=0.9, color="#AAAAAA", linestyle=":", linewidth=1, alpha=0.5)

    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_multiplicity_vs_performance(
    multiplicity_df: pl.DataFrame,
    performance_df: pl.DataFrame,
    output_path: str,
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """Plot correlation between multiplicity metrics and DTDG/CTDG performance.

    Args:
        multiplicity_df: DataFrame with columns: dataset, compression_ratio, etc.
        performance_df: DataFrame with columns: dataset, method, method_type, roc_auc.
        output_path: Path to save the plot.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Merge dataframes
    merged = performance_df.join(multiplicity_df, on="dataset", how="left")

    # Group by dataset and method_type, compute mean performance
    summary = (
        merged.group_by(["dataset", "method_type"])
        .agg(
            [
                pl.col("roc_auc").mean().alias("mean_roc_auc"),
                pl.col("compression_ratio").first(),
            ]
        )
        .sort("dataset")
    )

    for ax, (method_type, color, marker) in zip(
        axes, [("DTDG", COLORS["primary"], "o"), ("CTDG", COLORS["tertiary"], "s")]
    ):
        data = summary.filter(
            (pl.col("method_type") == method_type)
            & pl.col("compression_ratio").is_not_null()
            & pl.col("mean_roc_auc").is_not_null()
        )

        if not data.is_empty():
            x = data["compression_ratio"].to_numpy()
            y = data["mean_roc_auc"].to_numpy()
            labels = data["dataset"].to_list()

            ax.scatter(
                x,
                y,
                s=120,
                alpha=0.8,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidth=1.5,
                zorder=5,
            )

            # Add trend line if enough points
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x) * 0.95, max(x) * 1.05, 100)
                ax.plot(
                    x_line,
                    p(x_line),
                    "--",
                    color=color,
                    alpha=0.4,
                    linewidth=1.5,
                    zorder=1,
                )

            # Annotations with adjustable text
            for xi, yi, label in zip(x, y, labels):
                ax.annotate(
                    label.replace("-", "\n"),
                    (xi, yi),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=8,
                    ha="left",
                    va="center",
                )

        ax.set_xlabel("Compression Ratio", fontweight="medium")
        ax.set_ylabel("Mean ROC-AUC", fontweight="medium")
        ax.set_title(f"{method_type} Methods", fontweight="bold")

    fig.suptitle(
        "Edge Multiplicity vs Detection Performance",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_correlation_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    output_path: str,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Correlation",
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Create scatter plot with labels for correlation analysis.

    Args:
        x: X-axis values.
        y: Y-axis values.
        labels: Point labels (e.g., dataset names).
        output_path: Path to save the plot.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x,
        y,
        s=120,
        alpha=0.8,
        color=COLORS["primary"],
        edgecolors="white",
        linewidth=1.5,
        zorder=5,
    )

    # Annotations
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(
            label.replace("-", "\n"),
            (xi, yi),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="center",
        )

    # Add trend line and statistics
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x) * 0.95, max(x) * 1.05, 100)
        ax.plot(
            x_line,
            p(x_line),
            "--",
            color=COLORS["secondary"],
            alpha=0.6,
            linewidth=2,
            zorder=1,
        )

        # Compute correlation
        corr = np.corrcoef(x, y)[0, 1]
        stats_text = f"r = {corr:.3f}\ny = {z[0]:.3f}x + {z[1]:.3f}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.9,
            },
        )

    ax.set_xlabel(xlabel, fontweight="medium")
    ax.set_ylabel(ylabel, fontweight="medium")
    ax.set_title(title)

    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_multiplicity_distribution(
    distribution: dict[str, int],
    output_path: str,
    dataset: str,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Plot bar chart of edge multiplicity distribution.

    Args:
        distribution: Dictionary mapping multiplicity to count.
        output_path: Path to save the plot.
        dataset: Dataset name for title.
        figsize: Figure size as (width, height).
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort keys, handling "4+" specially
    keys = sorted([k for k in distribution.keys() if k != "4+"])
    if "4+" in distribution:
        keys.append("4+")

    values = [distribution[k] for k in keys]
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    # Create bars with gradient-like coloring
    colors = [COLORS["primary"] if k == "1" else COLORS["tertiary"] for k in keys]
    bars = ax.bar(
        range(len(keys)), values, color=colors, edgecolor="white", linewidth=1
    )

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_xlabel("Edge Multiplicity", fontweight="medium")
    ax.set_ylabel("Count", fontweight="medium")
    ax.set_title(f"Edge Multiplicity Distribution\n{dataset.replace('-', ' ').title()}")

    # Add percentage labels on top of bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="medium",
        )

    # Add summary statistics
    stats_text = (
        f"Total edges: {total:,}\nUnique: {values[0]:,} ({percentages[0]:.1f}%)"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.9,
        },
    )

    plt.tight_layout()
    _save_figure(fig, output_path)


def create_summary_table(
    spectral_results: dict[str, dict[str, Any]],
    multiplicity_results: dict[str, dict[str, Any]],
    output_path: str,
) -> None:
    """Create and save a summary table of analysis results.

    Args:
        spectral_results: Dict mapping dataset to spectral metrics.
        multiplicity_results: Dict mapping dataset to multiplicity stats.
        output_path: Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    for dataset in sorted(
        set(spectral_results.keys()) | set(multiplicity_results.keys())
    ):
        spectral = spectral_results.get(dataset, {})
        mult = multiplicity_results.get(dataset, {})

        record = {
            "dataset": dataset,
            "num_nodes": spectral.get("num_nodes"),
            "num_edges": mult.get("total_edges"),
            "unique_edges": mult.get("unique_edges"),
            "compression_ratio": mult.get("compression_ratio"),
            "pct_repeated": mult.get("pct_repeated"),
            "s_high": spectral.get("s_high"),
        }
        records.append(record)

    df = pl.DataFrame(records)
    df.write_csv(output_path)
