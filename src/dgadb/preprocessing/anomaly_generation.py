from math import gcd
from functools import reduce
from typing import Literal
import polars as pl
import torch
import numpy as np
import random


def generate_anomalies_no_timestamp(data: torch.Tensor, train_percent: float, anomaly_percent: float):
    """
    Generates a synthetic graph anomaly detection dataset from a list of edges. 
    Implements the method described in [Li Zheng et al., 2019], with reference to its public implementation.

    This function takes a complete set of graph edges and splits it into a
    training set and a testing set. Any columns beyond the first two in the input data
    (e.g., timestamps) are ignored.

    The final testing set is a shuffled mixture of:
    1.  Normal Edges: Edges from the original graph that were not included in
        the training set (label 0).
    2.  Anomalous Edges: Synthetically generated edges that connect nodes from
        the training graph but did not exist in the original graph (label 1).

    The number of anomalies is calculated to ensure that they constitute the
    specified `anomaly_percent` of the final, combined test set.

    Args:
        data (torch.Tensor): A 2D tensor of shape [n_edges, 2] representing
            the graph's edge list.
        train_percent (float): The percentage of edges (from the beginning of
            the `data` tensor) to be used for the training set. Must be
            between 0.0 and 1.0.
        anomaly_percent (float): The desired percentage of anomalous edges in the
            final testing set. Must be between 0.0 and 1.0.

    Returns:
        tuple:
            - train_edges (torch.Tensor): A tensor of edges for the training graph.
            - synthetic_test_edges (torch.Tensor): A 2D tensor of shape
              [n_test_edges, 3] representing the testing set. Each row is
              (source_node, target_node, label), where label is 0 for normal
              and 1 for anomalous.

    References:
        - Original paper: "AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based
Temporal GCN", Li Zheng et al., IJCAI, 2019.
        - Public implementation: https://github.com/Ljiajie/Addgraph/blob/master/UCI_D_Addgraph/framwork/anomaly_generation.py
    """
    if not 0.0 <= train_percent <= 1.0:
        raise ValueError("train_percent must be between 0.0 and 1.0")
    if not 0.0 <= anomaly_percent <= 1.0:
        raise ValueError("anomaly_percent must be between 0.0 and 1.0")

    all_edges = data[:, :2]
    n_edges = len(all_edges)
    n_train_edges = int(n_edges * train_percent)
    train_edges = all_edges[:n_train_edges]
    normal_test_edges = all_edges[n_train_edges:]

    train_nodes = torch.unique(train_edges).cpu()
    n_train = len(train_nodes)

    # Create a set of all existing undirected edges
    # An edge (u, v) is stored as (min(u,v), max(u,v)) to treat it as undirected
    canonical_edges, _ = torch.sort(all_edges, dim=1)
    unique_canonical_edges = torch.unique(canonical_edges, dim=0)
    canonical_edge_set = set(tuple(edge)
                             for edge in unique_canonical_edges.cpu().tolist())

    # Calculate the number of anomalies to generate
    n_normal_test_edges = len(normal_test_edges)
    n_anomalies = max(
        0, int((n_normal_test_edges * anomaly_percent) / (1 - anomaly_percent + 1e-10)))

    print(f"Training edges: {len(train_edges)}")
    print(f"Normal test edges: {n_normal_test_edges}")
    print(f"Anomalies to generate: {n_anomalies}\n")

    anomalies = []
    attempts = 0
    max_attempts = n_anomalies * 100 + 100  # Prevent infinite loops

    while len(anomalies) < n_anomalies and attempts < max_attempts:
        # Sample two distinct random training nodes
        sample_indices = torch.randperm(n_train)[:2]
        u, v = train_nodes[sample_indices].tolist()
        edge_candidate = (u, v)
        canonical_edge_candidate = tuple(sorted((u, v)))

        # Add the edge if it's not a self-loop and doesn't already exist
        if u != v and canonical_edge_candidate not in canonical_edge_set:
            anomalies.append(list(edge_candidate))
            # Add to set to prevent re-generating it
            canonical_edge_set.add(canonical_edge_candidate)
        else:
            attempts += 1

    if attempts >= max_attempts and len(anomalies) < n_anomalies:
        print(
            f"Warning: Could only generate {len(anomalies)} / {n_anomalies} requested anomalies.")

    anomaly_edges = torch.tensor(anomalies, dtype=torch.long)

    # Assemble the final test set
    if len(anomalies) > 0:
        test_edges = torch.cat([normal_test_edges, anomaly_edges], dim=0)
        # Create corresponding labels (0 for normal, 1 for anomaly)
        normal_labels = torch.zeros(n_normal_test_edges, dtype=torch.long)
        anomaly_labels = torch.ones(len(anomalies), dtype=torch.long)
        test_labels = torch.cat([normal_labels, anomaly_labels], dim=0)
        # Shuffle combined test set
        test_perm = torch.randperm(len(test_edges))
        test_edges = test_edges[test_perm]
        test_labels = test_labels[test_perm]
    else:  # Case with no anomalies
        test_edges = normal_test_edges
        test_labels = torch.zeros(n_normal_test_edges, dtype=torch.long)

    # Combine edges and labels into (source, target, label) format
    synthetic_test_edges = torch.cat(
        [test_edges, test_labels.unsqueeze(1)], dim=1)

    return train_edges, synthetic_test_edges


_REQUIRED_COLS = ["src", "tgt", "timestamp", "train_mask", "test_mask"]

_ANOMALY_TYPES = [
    "temporal",
    "contextual",
    "temporal-contextual",
    "structural-contextual",
    "temporal-structural-contextual"
]


class AnomalyGenerator:
    def __init__(
            self,
            edges: pl.DataFrame,
            anom_ratio: float,
            anom_type: Literal[
                "temporal",
                "contextual",
                "temporal-contextual",
                "structural-contextual",
                "temporal-structural-contextual",
                "combination"
            ]
    ) -> None:
        if not 0.0 <= anom_ratio <= 1.0:
            raise ValueError("'anomaly_percent' must be between 0.0 and 1.0.")

        missing = [col for col in _REQUIRED_COLS if col not in edges.columns]
        if missing:
            raise ValueError(f"Missing column(s): {missing}")

        self.edges = edges if edges["timestamp"].is_sorted() \
            else edges.sort("timestamp")

        # self.train_edges = edges.filter(pl.col("train_mask"))
        # self.test_edges = edges.filter(pl.col("test_mask"))
        # self.val_edges = (
        #     edges.filter(pl.col("val_mask"))
        #     if "val_mask" in edges.columns
        #     else None
        # )

        self.anom_ratio = anom_ratio
        self.anom_types = (
            _ANOMALY_TYPES
            if anom_type == "combination"
            else [anom_type]
        )

        self._build_edge_lookups()
        self._compute_temporal_properties()

    def _build_edge_lookups(self) -> None:
        unique_edges = self.edges.unique(subset=["src", "tgt"])
        unique_edges_ts = self.edges.unique(subset=["src", "tgt", "timestamp"])

        self.observed_edges = set(zip(
            unique_edges["src"],
            unique_edges["tgt"]
        ))
        self.observed_edges_with_timestamps = set(zip(
            unique_edges_ts["src"],
            unique_edges_ts["tgt"],
            unique_edges_ts["timestamp"]
        ))

    def _compute_temporal_properties(self) -> None:
        """
        Analyzes the training data to learn its temporal properties:
            1. The distribution of time deltas between events.
            2. The time granularity (e.g. 1 for every second, 60 for every minute).
        This should be called only once.
        """
        # Compute time deltas
        train_edges = self.edges.filter(pl.col("train_mask"))
        unique_timestamps = train_edges["timestamp"].unique().to_numpy()
        if len(unique_timestamps) < 2:
            self.time_deltas = np.array([], dtype=int)
        else:
            self.time_deltas = np.diff(unique_timestamps)

        # Compute granularity
        if len(self.time_deltas) == 0:
            self.granularity = 1
        else:
            # The granularity is the greatest common divisor of all time deltas
            self.granularity = reduce(gcd, self.time_deltas)

    # def _compute_edge_probabilities(self, edges: torch.Tensor) -> np.ndarray:
    #     """
    #     Computes edge probabilities with respect to their counts.
    #     More frequent edges have lower probability.

    #     Args:
    #         edges (torch.Tensor): A tensor of shape (n_edges, 2) where each row is a [source, destination] pair.

    #     Returns:
    #         torch.Tensor: A 1D tensor of shape (n_edges,) containing the probability for each edge
    #                     in the input tensor.
    #     """
    #     # 'inverse_indices' maps each edge in the original 'edges' tensor to its index in 'unique_edges'
    #     _, inverse_indices, counts = torch.unique(
    #         edges, dim=0, return_inverse=True, return_counts=True)
    #     # (1/counts) * (1/sqrt(counts))
    #     unique_weights = torch.pow(counts, -1.5)
    #     edge_weights = unique_weights[inverse_indices]
    #     return edge_weights / (edge_weights.sum() + 1e-10)

    def _compute_edge_probabilities(self, edges: pl.DataFrame) -> np.ndarray:
        """
        Computes edge probabilities with respect to their counts.
        More frequent edges have lower probability.

        Args:
            edges (pl.DataFrame): A Polars DataFrame with at least two columns 'src' and 'tgt'.

        Returns:
            np.ndarray: A 1D NumPy array of shape (n_edges,) containing the probability
                        for each edge in the input DataFrame, in the original order.
        """
        # Group by both source and target to find unique edges, then get the size of each group
        edge_counts = edges.group_by(['src', 'tgt']).agg(
            pl.len().alias('counts')
        )

        # (1/counts) * (1/sqrt(counts)) = counts ** -1.5
        unique_weights = edge_counts.with_columns(
            (pl.col('counts').cast(pl.Float64) ** -1.5).alias('weight')
        )

        # Join operation to assign weights to edges, does not guarantee to preserve order
        # Add a temporary column with row indices to ensure we can restore the original order
        edges_with_weights = edges.with_row_index().join(
            unique_weights.select(['src', 'tgt', 'weight']),
            on=['src', 'tgt'],
            how='left'
        )
        sorted_weights = edges_with_weights.sort('index')['weight']

        edge_weights = sorted_weights.to_numpy()
        total_weight = edge_weights.sum()

        return edge_weights / (total_weight + 1e-10)

    def _generate_plausible_timestamp(self, first_t: int, last_t: int, random_time_walk_max_step: int = 11) -> int:
        # Find the first possible base point at or after 'first_t'
        start_base_point = (first_t + self.granularity - 1) // self.granularity
        # Find the last possible base point at or before 'last_t'
        end_base_point = last_t // self.granularity

        if start_base_point > end_base_point:
            return first_t

        # Sample a random base point and convert it back to a timestamp
        random_base_point = np.random.randint(
            start_base_point, end_base_point + 1)
        start_t = random_base_point * self.granularity

        # Perform the random time walk
        # Fallback if no deltas
        if len(self.time_deltas) == 0:
            return start_t

        current_t = float(start_t)
        num_steps = np.random.randint(1, random_time_walk_max_step)
        for _ in range(num_steps):
            random_time_delta = np.random.choice(self.time_deltas)
            # 50% chance to add or subtract the delta
            if np.random.rand() < 0.5:
                current_t += random_time_delta
            else:
                current_t -= random_time_delta

        return int(np.clip(current_t, first_t, last_t))

    def _generate_anomalous_samples(self, use_val_split: bool = False):
        if use_val_split:
            if "val_mask" not in self.edges.columns:
                raise ValueError("'val_mask' not found.")
            eval_edges = self.edges.filter(pl.col("val_mask"))
        else:
            eval_edges = self.edges.filter(pl.col("test_mask"))

        num_anomalies = int(len(eval_edges) * self.anom_ratio)

        first_t = eval_edges["timestamp"].first()
        last_t = eval_edges["timestamp"].last()

        eval_edge_p = self._compute_edge_probabilities(eval_edges)

        for _ in range(num_anomalies):
            anom_type = random.choice(self.anom_types)

            if anom_type == "temporal":
                self._generate_one_temporal_anomaly(
                    eval_edges, first_t, last_t, eval_edge_p)

            elif anom_type == "contextual":
                pass

            elif anom_type == "temporal-contextual":
                pass

            elif anom_type == "structural-contextual":
                pass

            elif anom_type == "temporal-structural-contextual":
                pass

    def _generate_one_temporal_anomaly(self, eval_edges: pl.DataFrame, first_t: int, last_t: int, edge_p: np.ndarray, max_num_tries: int = 100):
        ind = np.random.choice(len(self.edges), 1, p=edge_p)
        anom_src, anom_tgt, _ = eval_edges.row(ind)
        anom_t = None
        num_tries = 0
        while (
            anom_t is None
            or (anom_src, anom_tgt, anom_t) in self.observed_edges_with_timestamps
            and (num_tries < max_num_tries)
        ):
            anom_t = self._generate_plausible_timestamp(first_t, last_t)
            num_tries += 1

        return anom_src, anom_tgt, anom_t
