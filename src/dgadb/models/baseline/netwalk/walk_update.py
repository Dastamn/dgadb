import logging
from typing import Generator

import torch
import numpy as np

from .reservoir import Reservoir


class WalkUpdate:
    def __init__(
            self,
            train_edges: torch.Tensor,
            total_num_nodes: int,
            num_walks: int,
            walk_length: int,
            reservoir_dim: int = 10,
            walk_restart_prob: float = 0.0,
            prev_walk_ratio: float = 0.1,
            seed: int = 1234
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train_edges = train_edges
        self.total_num_nodes = total_num_nodes
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walk_restart_prob = walk_restart_prob
        self.prev_train_ratio = prev_walk_ratio
        self._rng = np.random.RandomState(seed)

        self.reservoir = Reservoir(
            train_edges, total_num_nodes, reservoir_dim, seed)

        self.walks = self.__init_walks()
        self.new_walks = None

    def __init_walks(self) -> list[list]:
        walks = []
        nodes = np.asarray(self.reservoir.get_nodes())
        for _ in range(self.num_walks):
            self._rng.shuffle(nodes)
            for u in nodes:
                walks.append(self.__random_walk(u, self.walk_restart_prob))
        return walks

    def __random_walk(self, start_node: int, restart_prob: float = 0.0) -> list:
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = list(
                self.reservoir.get_node_neighborhood(current_node))
            if len(neighbors) > 0:
                if self._rng.random() >= restart_prob:
                    walk.append(self._rng.choice(neighbors))
                else:
                    walk.append(walk[0])
            else:
                # padding when no neighbors
                remaining_length = self.walk_length - len(walk)
                walk.extend([current_node] * remaining_length)
                break
        return walk

    def __update_walks(self, new_edges: torch.Tensor) -> None:
        walk_list = []
        start_node = set()
        for u, v in new_edges:
            start_node.add(u.item())
            start_node.add(v.item())
        for s in set(start_node):
            for _ in range(self.num_walks):
                u = self._rng.choice(self.reservoir.get_reservoir(s))
                v = self._rng.choice(self.reservoir.get_reservoir(u))
                walk_list.append([s, u, v])

        self.new_walks = walk_list
        old_walks = [w for w in self.walks if w[0] not in start_node]
        self.walks = old_walks + self.new_walks

    def get_one_hot(self, walks: list[list]) -> torch.Tensor:
        walk_mat = torch.tensor(walks, dtype=torch.long)
        row_indices = walk_mat.flatten()
        col_indices = torch.arange(len(row_indices), dtype=torch.long)
        values = torch.ones(len(row_indices), dtype=torch.float32)
        indices = torch.stack([row_indices, col_indices])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(
            (self.total_num_nodes, len(walks) * self.walk_length)))
        return sparse_tensor.to_dense()

    def get_one_hot_walks(self) -> torch.Tensor:
        return self.get_one_hot(self.walks)

    def get_one_hot_walks_batch(self, batch_size: int) -> Generator[torch.Tensor, None, None]:
        num_walks = len(self.walks)
        for start in range(0, num_walks, batch_size):
            end = start + batch_size
            batch_walks = self.walks[start:end]
            yield self.get_one_hot(batch_walks)

    def update(self, new_edges: torch.Tensor) -> None:
        self.reservoir.update(new_edges)
        self.__update_walks(new_edges)
