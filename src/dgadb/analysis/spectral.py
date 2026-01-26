"""Spectral analysis functions for temporal graphs.

This module provides functions for computing spectral properties of graphs,
including Laplacian matrices, eigenvalues, and spectral density metrics.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

from dgadb.storage import TemporalGraph


def compute_normalized_laplacian(adj: csr_matrix) -> csr_matrix:
    """Compute the normalized Laplacian of a graph.

    Args:
        adj: Sparse adjacency matrix in CSR format.

    Returns:
        Normalized Laplacian matrix L = I - D^{-1/2} A D^{-1/2}.
    """
    return laplacian(adj, normed=True)


def compute_eigenvalues(
    L: csr_matrix,
    k: int | None = None,
    return_eigenvectors: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the smallest eigenvalues of a Laplacian matrix.

    For small graphs (n < 500), computes full decomposition.
    For larger graphs, computes k smallest eigenvalues using sparse solver.

    Args:
        L: Normalized Laplacian matrix in CSR format.
        k: Number of eigenvalues to compute. If None, uses min(200, n-2).
        return_eigenvectors: If True, also return eigenvectors.

    Returns:
        Eigenvalues sorted in ascending order.
        If return_eigenvectors is True, also returns eigenvector matrix.
    """
    n = L.shape[0]

    if n < 500:
        # Full eigenvalue decomposition for small graphs
        L_dense = L.toarray()
        if return_eigenvectors:
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            return eigenvalues, eigenvectors
        return np.linalg.eigvalsh(L_dense)

    # Sparse decomposition for larger graphs
    if k is None:
        k = min(200, n - 2)

    # Ensure k is valid
    k = min(k, n - 2)
    if k <= 0:
        if return_eigenvectors:
            return np.array([]), np.array([]).reshape(n, 0)
        return np.array([])

    # Use eigsh with 'SM' (smallest magnitude) for Laplacian eigenvalues
    # The smallest eigenvalue of a connected graph Laplacian is 0
    if return_eigenvectors:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    eigenvalues = eigsh(L, k=k, which="SM", return_eigenvectors=False)
    return np.sort(eigenvalues)


def get_binary_adjacency(
    tg: TemporalGraph,
    mask: torch.Tensor | None = None,
) -> csr_matrix:
    """Get binary adjacency matrix from temporal graph with deduplication.

    Creates an undirected binary adjacency matrix where each unique (src, tgt)
    pair gets a single edge with value 1. This represents what DTDG methods see.

    Args:
        tg: Temporal graph to extract adjacency from.
        mask: Optional boolean mask to filter edges.

    Returns:
        Binary CSR adjacency matrix.
    """
    src = tg.src.cpu().numpy()
    tgt = tg.tgt.cpu().numpy()

    if mask is not None:
        mask_np = mask.cpu().numpy()
        src = src[mask_np]
        tgt = tgt[mask_np]

    # Create undirected edges by adding both directions
    # Use set to deduplicate
    edge_set = set()
    for s, t in zip(src, tgt):
        edge_set.add((s, t))
        edge_set.add((t, s))  # Make undirected

    if not edge_set:
        n = tg.num_nodes
        return csr_matrix((n, n), dtype=np.float32)

    edges = np.array(list(edge_set))
    rows = edges[:, 0]
    cols = edges[:, 1]
    data = np.ones(len(rows), dtype=np.float32)

    n = tg.num_nodes
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def compute_degree_signal(adj: csr_matrix) -> np.ndarray:
    """Compute degree-based node signal from adjacency matrix.

    Args:
        adj: Sparse adjacency matrix.

    Returns:
        1D array of node degrees.
    """
    return np.asarray(adj.sum(axis=1)).flatten()


def compute_s_high(L: csr_matrix, signal: np.ndarray) -> float:
    """Compute the high-frequency energy (S_high) using Rayleigh quotient.

    S_high = signal^T @ L @ signal / (signal^T @ signal)

    This measures the smoothness of the signal on the graph. Higher values
    indicate the signal has more high-frequency content (less smooth).

    Args:
        L: Normalized Laplacian matrix.
        signal: Node signal as 1D array.

    Returns:
        S_high value (Rayleigh quotient).
    """
    signal = signal.astype(np.float64)
    numerator = signal @ (L @ signal)
    denominator = signal @ signal

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def compute_spectral_density(
    eigenvalues: np.ndarray,
    bins: int = 50,
    range_min: float = 0.0,
    range_max: float = 2.0,
) -> dict:
    """Compute histogram of eigenvalue distribution.

    For normalized Laplacian, eigenvalues are in [0, 2].

    Args:
        eigenvalues: Array of eigenvalues.
        bins: Number of histogram bins.
        range_min: Minimum value for histogram range.
        range_max: Maximum value for histogram range.

    Returns:
        Dictionary with 'counts', 'bin_edges', and 'bin_centers'.
    """
    counts, bin_edges = np.histogram(
        eigenvalues, bins=bins, range=(range_min, range_max)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
    }


def compute_energy_ratio_curve(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative energy ratio at different frequency thresholds.

    For each eigenvalue threshold, computes the fraction of signal energy
    captured by eigenvectors with eigenvalues below that threshold.

    Args:
        eigenvalues: Sorted eigenvalues (ascending).
        eigenvectors: Eigenvector matrix where column i corresponds to eigenvalue i.
        signal: Node signal as 1D array.

    Returns:
        Tuple of (thresholds, cumulative_energy_ratios).
        thresholds: Eigenvalue thresholds.
        cumulative_energy_ratios: Fraction of energy at each threshold.
    """
    if len(eigenvalues) == 0:
        return np.array([]), np.array([])

    signal = signal.astype(np.float64)
    total_energy = signal @ signal

    if total_energy < 1e-10:
        return eigenvalues, np.ones_like(eigenvalues)

    # Project signal onto each eigenvector
    # coefficients[i] = <signal, v_i>^2
    coefficients = (eigenvectors.T @ signal) ** 2

    # Cumulative sum of coefficients gives energy up to each frequency
    cumulative_energy = np.cumsum(coefficients)
    energy_ratios = cumulative_energy / total_energy

    return eigenvalues, energy_ratios
