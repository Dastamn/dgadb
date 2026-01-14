from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from dataclasses import dataclass
import logging
from collections import defaultdict
from typing import Self

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.sparse import csgraph

from src.dgadb.experiment.callbacks import ExperimentCallback, ExperimentCallbackHandler
from src.dgadb.storage.temporal_graph import TemporalGraph
from src.dgadb.storage.temporal_snapshot import TemporalGraphSnapshotLoader

from ..base import BaseADModel, BaseADModelComponents, TrainingState
from src.dgadb.storage import TemporalGraphSnapshot, TemporalGraphLoader


class Reservoir:
    def __init__(
            self,
            train_edges: torch.Tensor,
            # train_neighborhoods: dict,
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
        self.train_neighborhoods = defaultdict(set)
        self._reservoir = defaultdict(
            lambda: np.full(self.dim, -1, dtype=np.int64))

        self.__update_node_neighborhoods(self.train_edges)
        self.__build_reservoir()

    def __update_node_neighborhood(self, u: int, v: int) -> None:
        if u != v:
            self.train_neighborhoods[u].add(v)
            self.train_neighborhoods[v].add(u)

    def __update_node_neighborhoods(self, new_edges: torch.Tensor) -> None:
        _ = [self.__update_node_neighborhood(int(u.item()), int(v.item()))
             for u, v in new_edges]

    def __build_reservoir(self) -> None:
        for u in range(self.total_num_nodes):
            neighbor_list = list(self.train_neighborhoods[u])
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

    def update(self, new_edges: torch.Tensor):  # shape: (m, 2)
        self.__update_node_neighborhoods(new_edges)
        for u, v in new_edges:
            u = int(u.item())
            v = int(v.item())
            self.__update_reservoir(u, v)
            self.__update_reservoir(v, u)


class WalkUpdate:
    def __init__(
            self,
            train_edges: torch.Tensor,
            # train_neighborhoods: dict,
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
        self.train_neighborhoods = train_neighborhoods
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walk_restart_prob = walk_restart_prob
        self.prev_train_ratio = prev_walk_ratio
        self._rng = np.random.RandomState(seed)

        self.reservoir = Reservoir(
            train_edges, train_neighborhoods, total_num_nodes, reservoir_dim, seed)

        self.walks = self.__init_walks()
        self.new_walks = None
        # training_walks = new_walks + "percent" * old_walks
        # self.training_walks = None

    def __init_walks(self) -> list[list]:
        walks = []
        nodes = np.asarray(list(self.train_neighborhoods.keys()))
        for _ in range(self.num_walks):
            self._rng.shuffle(nodes)
            for u in nodes:
                walks.append(self.__random_walk(u, self.walk_restart_prob))
        return walks

    def __random_walk(self, start_node: int, restart_prob: float = 0.0) -> list:
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = list(self.train_neighborhoods[current_node])
            if len(neighbors) > 0:
                if self._rng.random() >= restart_prob:
                    walk.append(self._rng.choice(neighbors))
                else:
                    walk.append(walk[0])
            else:
                break
        return walk

    def update_node_neighborhood(self, new_edges: torch.Tensor) -> None:
        for u, v in self.train_edges:
            if u != v:
                u = u.item()
                v = v.item()
                self.train_neighborhoods[u].add(v)
                self.train_neighborhoods[v].add(u)

    def update_reservoir(self, new_edges: torch.Tensor) -> None:
        walk_list = []
        start_node = set()
        for u, v in new_edges:
            start_node.add(u.item())
            start_node.add(v.item())
        for s in set(start_node):
            for n in range(self.num_walks):
                u = self._rng.choice(self.reservoir.get_reservoir(s))
                v = self._rng.choice(self.reservoir.get_reservoir(u))
                walk_list.append([s, u, v])

        self.new_walks = walk_list
        old_walks = [w for w in self.walks if w[0] not in start_node]
        # self.training_walks = old_walks + self.new_walks
        self.walks = old_walks + self.new_walks
        self.logger.info(
            f"Number of previous walks: {len(self.walks)}")

    def get_one_hot(self, walks: list[list]) -> torch.Tensor:
        walk_mat = torch.tensor(walks, dtype=torch.long)
        row_indices = walk_mat.flatten()
        col_indices = torch.arange(len(row_indices), dtype=torch.long)
        values = torch.ones(len(row_indices), dtype=torch.float32)
        indices = torch.stack([row_indices, col_indices])
        sparse_tensor = torch.sparse_coo_tensor(indices, values)
        return sparse_tensor.to_dense()

    def get_one_hot_walks(self) -> torch.Tensor:
        return self.get_one_hot(self.walks)

    # def get_init_walk(self) -> torch.Tensor:
    #     return self.get_one_hot(self.walks)


class NetwalkAE(nn.Module):
    def __init__(
        self,
        in_channels: int,  # num_nodes
        hidden_channels: int
    ) -> None:
        super().__init__()
        self.encoder_layer = nn.Linear(in_channels, hidden_channels, bias=True)
        self.decoder_layer = nn.Linear(in_channels, hidden_channels, bias=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder_layer.weight, gain=1.0)
        nn.init.xavier_uniform_(self.decoder_layer.weight, gain=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # (batch_size, num_nodes)
        return torch.sigmoid(self.encoder_layer(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder_layer(h))

    def forward(self, x: torch.Tensor, corrupt_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
        x_corrupted = (
            (x + torch.rand_like(x)) * corrupt_prob + x * (1 - corrupt_prob)
            if self.training
            else x
        )
        h = self.encode(x_corrupted)
        x_hat = self.decode(h)
        return x_hat, h


@dataclass
class NetwalkComponents(BaseADModelComponents):
    autoencoder: NetwalkAE
    optimizer: torch.optim.Optimizer
    walk_update: WalkUpdate | None = None


class Netwalk(BaseADModel[NetwalkComponents]):
    def __init__(
        self,
        num_walks: int = 20,
        walk_length: int = 3,
        hidden_size: int = 20,  # node feature dim
        sparcity_ratio: float = 0.5,  # rho
        weight_decay: float = 0.0017,  # lambda
        sparcity_weight: float = 1.0,  # beta
        ae_weight: float = 340.0,  # gamma
        batch_size: int = 40,
        learning_rate: float = 0.01,
        corrupt_prob: float = 0.0,
        clustering_update: float = 0.01,  # alpha
        num_clusters: int = 3,  # k
        device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.hidden_size = hidden_size
        self.sparcity_ratio = sparcity_ratio
        self.weight_decay = weight_decay
        self.sparcity_weight = sparcity_weight
        self.ae_weight = ae_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.corrupt_prob = corrupt_prob
        self.clustering_update = clustering_update
        self.num_clusters = num_clusters

        self.train_neighborhoods = defaultdict(set)
        self._L = None

    def _compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = self.ae_weight * F.mse_loss(x_hat, x)

        # Sparcity loss (KL loss)
        z_sparcity_ratio = torch.mean(h, dim=0)
        z_sparcity_ratio = torch.clamp(z_sparcity_ratio, 1e-6, 1 - 1e-6)
        kl_div = (self.sparcity_ratio * torch.log(self.sparcity_ratio / z_sparcity_ratio)) + \
            ((1 - self.sparcity_ratio) *
             torch.log((1 - self.sparcity_ratio) / (1 - z_sparcity_ratio)))

        sparsity_loss = self.sparcity_weight * torch.sum(kl_div)

        # Clique loss
        # z shape: (batch_size, hidden_size)
        # batch_size = num_walks * walk_length
        num_walks = h.shape[0] // self.walk_length
        h_reshaped = h.view(num_walks, self.walk_length, self.hidden_size)

        # trace(H^T * L * H)
        # M = L @ H
        M = torch.einsum('ij, wjk -> wik', self._L, h_reshaped)
        trace_per_walk = torch.einsum('wld, wld -> w', h_reshaped, M)
        clique_loss = trace_per_walk.mean()

        return recon_loss + sparsity_loss + clique_loss

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        ae = self.components.autoencoder
        optimizer = self.components.optimizer
        walk_update = self.components.walk_update

        current_graph = snapshot.current
        current_edges = current_graph.edges

        is_first_snapshot = kwargs.get("is_first_snapshot", False)
        # if is_first_snapshot:
        #     walk_update.

        # 1. Update the walk generator with new edges from the snapshot
        # We skip this for the first snapshot because the walks are already initialized.
        if not is_first_snapshot:
            # Note: A snapshot might contain edges from a certain time window, not just new ones.
            # Assuming snapshot.edge_index.T contains the new edges for this step.
            new_edges = snapshot.edge_index.T
            walk_update.update(new_edges)

        # 2. Get the new, complete set of one-hot walks for training
        # This data represents the cumulative graph state.
        # Shape: (num_nodes, total_walk_len)
        walk_data = walk_update.get_one_hot_walks()

        # 3. Create a DataLoader for the walk data.
        # We transpose the data so that each item in the dataset is a one-hot vector for a single step in a walk.
        # Shape of walk_data.T: (total_walk_len, num_nodes)
        train_dataset = TensorDataset(walk_data.T)
        data_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        # 4. Train the autoencoder from scratch for N epochs on this new data
        epoch_losses = []
        for epoch in range(epochs):
            batch_losses = []
            for (batch,) in data_loader:  # (batch,) unpacks the single tensor from TensorDataset
                batch = batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                x_hat, h = ae(batch, self.corrupt_prob)

                # Compute loss
                loss = self._compute_loss(batch, x_hat, h)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        # Return the average loss from the final epoch
        return np.mean(epoch_losses)

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        num_nodes = data.num_nodes
        if self.batch_size > (size := self.num_walks * num_nodes):
            raise ValueError("`batch_size` should be smaller or equal to `number_walks * num_nodes`, "
                             f"found: {self.batch_size} > {size}")

        train_edges = data.edge_index.T[data.train_mask, :]
        for u, v in tqdm(train_edges, desc="Locating Neighbors"):
            if u != v:
                self.train_neighborhoods[u].add(v)
                self.train_neighborhoods[v].add(u)

        walk_update = WalkUpdate(
            train_edges, self.train_neighborhoods, num_nodes, self.num_walks, self.walk_length)

        device = self.device

        ae = NetwalkAE(num_nodes, self.hidden_size)
        optimizer = torch.optim.Adam(
            ae.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        phi = np.ones((self.walk_length, self.walk_length)) - \
            np.eye(self.walk_length)
        self._L = torch.tensor(csgraph.laplacian(
            phi, normed=False), dtype=torch.float32, device=device)

        self._components = NetwalkComponents(
            walk_update, ae, optimizer).to(device)

    def train(self, epochs: int, train_loader: TemporalGraphSnapshotLoader, val_loader: TemporalGraphSnapshotLoader | None = None, callbacks: list[ExperimentCallback] | None = None):
        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        num_train_snapshots = len(train_loader)
        for i, train_snapshot in tqdm(enumerate(train_loader), total=num_train_snapshots, desc=f"Train Snapshots"):
            self.set_training_mode(True)
            cumulative_graph = train_snapshot.cumulative
            if cumulative_graph is None:
                raise ValueError()

            data = Data(edge_index=cumulative_graph.edge_index)
            data_loader = DataLoader(
                [data], batch_size=self.batch_size, shuffle=True)

            for epoch in tqdm(range(epochs), desc=f"Train Snapshot {i+1}/{num_train_snapshots}"):

                for batch in data_loader:
                    pass

                state.step_in_epoch = i
                state.total_steps += 1
                handler.on_train_step_begin(state)
                state.loss = self._train_step(train_snapshot)
                handler.on_train_step_end(state)

            if val_loader:
                self.set_training_mode(False)
                pass

    def _predict(self, snapshot: TemporalGraphSnapshot) -> torch.Tensor:
        return torch.empty()

    def save(self, save_dir: str) -> None:
        pass

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> Self:
        return cls()


if __name__ == "__main__":
    # loader = TemporalGraphLoader()
    # data = loader.load("bitcoin-alpha", "structural",
    #                    anom_test_ratio=0.2, create_if_not_found=True)
    # netwalk = Netwalk()
    # netwalk.setup(data)

    edges = torch.tensor([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 4],
        [0, 2],
    ])

    neighborhood_dict = defaultdict(set)
    for u, v in tqdm(edges, desc="Locating Neighbors"):
        if u != v:
            u, v = u.item(), v.item()
            neighborhood_dict[u].add(v)
            neighborhood_dict[v].add(u)

    walk_update = WalkUpdate(edges, neighborhood_dict,
                             5, num_walks=2, walk_length=3)
    prev_walks = walk_update.walks
