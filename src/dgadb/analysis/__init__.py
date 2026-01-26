"""Analysis modules for spectral analysis and multigraph experiments."""

from dgadb.analysis.spectral import (
    compute_normalized_laplacian,
    compute_eigenvalues,
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
)

__all__ = [
    # spectral
    "compute_normalized_laplacian",
    "compute_eigenvalues",
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
]
