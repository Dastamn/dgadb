import numpy as np
from dgadb.storage import TemporalGraph
from typing import Literal
import torch
from time import perf_counter
from functools import wraps
import logging
logger = logging.getLogger(__name__)


def time_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' took {total_time:.4f} seconds")
        return result
    return wrapper


class NeighborSampler:
    def __init__(
        self,
        temporal_graph: TemporalGraph,
        split: Literal["train", "val", "test"],
        sample_neighbor_strategy: Literal["uniform", "recent"] = "uniform",
        seed: int | None = None
    ) -> None:
        self.split = split
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

        self.__build_adj_list(temporal_graph, split)

    def __build_adj_list(self, temporal_graph: TemporalGraph, split: Literal["train", "val", "test"]):
        if split == "train":
            mask = temporal_graph.train_mask
        elif split == "val":
            mask = temporal_graph.train_mask | temporal_graph.val_mask
        elif split == "test":
            mask = temporal_graph.train_mask | temporal_graph.val_mask | temporal_graph.test_mask
        else:
            raise ValueError(f"Invalid split: {split}")

        if torch.sum(mask) == 0:
            raise ValueError(f"No values found for mask: {self.split}")

        src_nodes = temporal_graph.src[mask].cpu().numpy()
        tgt_nodes = temporal_graph.tgt[mask].cpu().numpy()
        timestamps = temporal_graph.t[mask].cpu().numpy()
        edge_indices = torch.arange(len(temporal_graph.src))[
            mask].cpu().numpy()

        num_nodes = temporal_graph.num_nodes
        # each entry will store a list of tuples: (neighbor_id, edge_idx, timestamp)
        adj_list = [[] for _ in range(num_nodes)]
        for src, tgt, edge_idx, t in zip(src_nodes, tgt_nodes, edge_indices, timestamps):
            adj_list[src].append((tgt, edge_idx, t))
            # adj_list[tgt].append((src, edge_idx, t))

        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        for node_idx in range(num_nodes):
            neighbors = adj_list[node_idx]
            if len(neighbors) == 0:
                self.nodes_neighbor_ids.append(np.array([], dtype=np.int64))
                self.nodes_edge_ids.append(np.array([], dtype=np.int64))
                self.nodes_neighbor_times.append(
                    np.array([], dtype=temporal_graph.t[0].dtype))
                continue

            # sort by timestamp
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])

            # Unpack the sorted tuples into separate numpy arrays
            neighbor_ids, edge_ids, neighbor_times = zip(*sorted_neighbors)

            self.nodes_neighbor_ids.append(
                np.array(neighbor_ids, dtype=np.int64))
            self.nodes_edge_ids.append(np.array(edge_ids, dtype=np.int64))
            self.nodes_neighbor_times.append(
                np.array(neighbor_times, dtype=np.float32))


if __name__ == "__main__":
    from dgadb.storage import TemporalGraphLoader
    loader = TemporalGraphLoader()
    tg = loader.load("bitcoin-alpha", create_if_not_found=True)
    sampler = NeighborSampler(tg, split="train")
