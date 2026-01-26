"""Analysis modules for spectral analysis and multigraph experiments."""

from dgadb.analysis.spectral import (
    compute_normalized_laplacian,
    compute_eigenvalues,
    compute_full_eigenvalues,
    compute_spectral_metrics,
    get_binary_adjacency,
    compute_degree_signal,
    compute_s_high,
    compute_spectral_density,
    compute_energy_ratio_curve,
)
from dgadb.analysis.multigraph import (
    MultiplicityStats,
    compute_multiplicity_stats,
    binary_deduplicate,
)
from dgadb.analysis.results_loader import load_experiment_results
from dgadb.analysis.visualization import (
    plot_eigenvalue_violin,
    plot_spectral_density,
    plot_delta_s_high_heatmap,
    plot_delta_s_high_grouped_bar,
    plot_energy_ratio_curves,
    plot_multiplicity_vs_performance,
    plot_correlation_scatter,
    plot_eigenvalue_kde_overlay,
    plot_eigenvalue_histogram_comparison,
    plot_delta_mean_heatmap,
    plot_frequency_band_bars,
    plot_concentration_at_one,
)

__all__ = [
    # spectral
    "compute_normalized_laplacian",
    "compute_eigenvalues",
    "compute_full_eigenvalues",
    "compute_spectral_metrics",
    "get_binary_adjacency",
    "compute_degree_signal",
    "compute_s_high",
    "compute_spectral_density",
    "compute_energy_ratio_curve",
    # multigraph
    "MultiplicityStats",
    "compute_multiplicity_stats",
    "binary_deduplicate",
    # results_loader
    "load_experiment_results",
    # visualization
    "plot_eigenvalue_violin",
    "plot_spectral_density",
    "plot_delta_s_high_heatmap",
    "plot_delta_s_high_grouped_bar",
    "plot_energy_ratio_curves",
    "plot_multiplicity_vs_performance",
    "plot_correlation_scatter",
    "plot_eigenvalue_kde_overlay",
    "plot_eigenvalue_histogram_comparison",
    "plot_delta_mean_heatmap",
    "plot_frequency_band_bars",
    "plot_concentration_at_one",
]
