"""Multigraph analysis functions for temporal graphs.

This module provides functions for analyzing edge multiplicity in temporal
graphs, which is important for understanding the difference between how
DTDG and CTDG methods perceive the graph structure.
"""

from dataclasses import dataclass, asdict
from collections import Counter

import numpy as np
import torch
from scipy.sparse import csr_matrix

from dgadb.storage import TemporalGraph


@dataclass
class MultiplicityStats:
    """Statistics about edge multiplicity in a temporal graph."""

    total_edges: int
    unique_edges: int
    compression_ratio: float  # total / unique
    pct_repeated: float  # percentage of edges that are duplicates
    max_multiplicity: int
    mean_multiplicity: float
    std_multiplicity: float
    multiplicity_distribution: dict[int, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_multiplicity_stats(
    tg: TemporalGraph,
    mask: torch.Tensor | None = None,
) -> MultiplicityStats:
    """Compute edge multiplicity statistics for a temporal graph.

    Groups edges by undirected (min(src,tgt), max(src,tgt)) pairs and
    counts how many times each edge appears.

    Args:
        tg: Temporal graph to analyze.
        mask: Optional boolean mask to filter edges.

    Returns:
        MultiplicityStats with edge repetition statistics.
    """
    src = tg.src.cpu().numpy()
    tgt = tg.tgt.cpu().numpy()

    if mask is not None:
        mask_np = mask.cpu().numpy()
        src = src[mask_np]
        tgt = tgt[mask_np]

    total_edges = len(src)

    if total_edges == 0:
        return MultiplicityStats(
            total_edges=0,
            unique_edges=0,
            compression_ratio=1.0,
            pct_repeated=0.0,
            max_multiplicity=0,
            mean_multiplicity=0.0,
            std_multiplicity=0.0,
            multiplicity_distribution={},
        )

    # Create canonical edge representation (min, max)
    edge_tuples = [(min(s, t), max(s, t)) for s, t in zip(src, tgt)]

    # Count multiplicities
    edge_counts = Counter(edge_tuples)
    unique_edges = len(edge_counts)
    multiplicities = list(edge_counts.values())

    compression_ratio = total_edges / unique_edges if unique_edges > 0 else 1.0
    # Percentage of edges that are duplicates (i.e., not the first occurrence)
    pct_repeated = (
        ((total_edges - unique_edges) / total_edges * 100) if total_edges > 0 else 0.0
    )

    # Multiplicity distribution
    mult_counts = Counter(multiplicities)
    # Group 4+ together
    distribution = {}
    for mult, count in sorted(mult_counts.items()):
        if mult >= 4:
            distribution["4+"] = distribution.get("4+", 0) + count
        else:
            distribution[str(mult)] = count

    return MultiplicityStats(
        total_edges=total_edges,
        unique_edges=unique_edges,
        compression_ratio=compression_ratio,
        pct_repeated=pct_repeated,
        max_multiplicity=max(multiplicities) if multiplicities else 0,
        mean_multiplicity=float(np.mean(multiplicities)) if multiplicities else 0.0,
        std_multiplicity=float(np.std(multiplicities)) if multiplicities else 0.0,
        multiplicity_distribution=distribution,
    )


def binary_deduplicate(
    tg: TemporalGraph,
    mask: torch.Tensor | None = None,
) -> csr_matrix:
    """Create binary adjacency matrix with deduplicated edges.

    This represents what DTDG methods see: an undirected binary graph
    where repeated edges are collapsed into single edges.

    Args:
        tg: Temporal graph to deduplicate.
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

    # Create undirected edges and deduplicate
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
