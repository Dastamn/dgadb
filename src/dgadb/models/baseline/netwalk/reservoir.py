import logging
from collections import defaultdict

import torch
import numpy as np


class Reservoir:
    def __init__(
            self,
            train_edges: torch.Tensor,
            total_num_nodes: int,
            dim: int = 10,
            seed: int = 1234
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._rng = np.random.RandomState(seed)

        self.train_edges = train_edges
        self.total_num_nodes = total_num_nodes
        self.dim = dim

        self._degrees = defaultdict(int)
        self.node_neighborhoods = defaultdict(set)
        self._reservoir = defaultdict(
            lambda: np.full(self.dim, -1, dtype=np.int64))

        self.__update_node_neighborhoods(self.train_edges)
        self.__build_reservoir()

    def __update_node_neighborhood(self, u: int, v: int) -> None:
        self.node_neighborhoods[u].add(v)

    def __update_node_neighborhoods(self, new_edges: torch.Tensor) -> None:
        for u, v in new_edges:
            u = int(u.item()) if isinstance(u, torch.Tensor) else u
            v = int(v.item()) if isinstance(v, torch.Tensor) else v
            self.__update_node_neighborhood(u, v)
            self.__update_node_neighborhood(v, u)

    def __build_reservoir(self) -> None:
        for u in range(self.total_num_nodes):
            neighbor_list = list(self.node_neighborhoods[u])
            degree = len(neighbor_list)
            if degree > 0:
                self._degrees[u] = degree
                idx = self._rng.randint(degree, size=self.dim)
                self._reservoir[u] = np.array(
                    [neighbor_list[i] for i in idx], dtype=np.int64)
            else:
                self._reservoir[u] = np.full(self.dim, -1, dtype=np.int64)

    def __update_reservoir(self, u: int, v: int):
        if not np.isin(v, self._reservoir[u]):
            self._degrees[u] += 1
            idx = self._rng.randint(self._degrees[u], size=self.dim)
            replace_idx = np.where(idx == self._degrees[u] - 1)
            self._reservoir[u][replace_idx] = v

    def get_reservoir(self, u: int) -> np.ndarray:
        return self._reservoir[u]

    def get_node_neighborhood(self, u: int) -> set:
        return self.node_neighborhoods[u]

    def get_nodes(self) -> list:
        return list(self.node_neighborhoods)

    def update(self, new_edges: torch.Tensor):
        for u, v in new_edges:
            u = int(u.item())
            v = int(v.item())
            self.__update_node_neighborhood(u, v)
            self.__update_node_neighborhood(v, u)
            self.__update_reservoir(u, v)
            self.__update_reservoir(v, u)
