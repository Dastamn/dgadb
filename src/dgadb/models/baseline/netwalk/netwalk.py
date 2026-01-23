from dataclasses import dataclass
from collections import Counter
from typing import Self
from math import ceil

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.sparse import csgraph
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

from dgadb.experiment.callbacks import ExperimentCallback, ExperimentCallbackHandler
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshotLoader

from .walk_update import WalkUpdate
from .autoencoder import Autoencoder
from dgadb.models.base import BaseADModel, BaseADModelComponents, TrainingState
from dgadb.storage import TemporalGraphSnapshot, TemporalGraphLoader


@dataclass
class NetwalkComponents(BaseADModelComponents):
    autoencoder: Autoencoder
    optimizer: torch.optim.Optimizer
    L: torch.Tensor | None = None
    walk_update: WalkUpdate | None = None
    kmeans: KMeans | None = None


class Netwalk(BaseADModel[NetwalkComponents]):
    def __init__(
        self,
        num_walks: int = 20,
        walk_length: int = 3,
        reservoir_dim: int = 10,
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
        self.reservoir_dim = reservoir_dim
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

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        num_nodes = data.num_nodes
        if self.batch_size > (size := self.num_walks * num_nodes):
            raise ValueError("`batch_size` should be smaller or equal to `number_walks * num_nodes`, "
                             f"found: {self.batch_size} > {size}")

        ae = Autoencoder(num_nodes, self.hidden_size)
        optimizer = torch.optim.Adam(
            ae.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        phi = np.ones((self.walk_length, self.walk_length)) - \
            np.eye(self.walk_length)
        L = torch.tensor(csgraph.laplacian(
            phi, normed=False), dtype=torch.float32)

        self._components = NetwalkComponents(
            autoencoder=ae, optimizer=optimizer, L=L).to(self.device)

    def _compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # print(f"x: {x.shape}, x_hat: {x_hat.shape}, h: {h.shape}")
        # Reconstruction loss
        recon_loss = self.ae_weight * F.mse_loss(x_hat, x)

        # Sparcity loss (KL loss)
        h_sparcity_ratio = torch.mean(h, dim=0)
        h_sparcity_ratio = torch.clamp(h_sparcity_ratio, 1e-6, 1 - 1e-6)
        kl_div = (self.sparcity_ratio * torch.log(self.sparcity_ratio / h_sparcity_ratio)) + \
            ((1 - self.sparcity_ratio) *
             torch.log((1 - self.sparcity_ratio) / (1 - h_sparcity_ratio)))

        sparsity_loss = self.sparcity_weight * torch.mean(kl_div)

        # Clique loss
        num_walks = h.shape[0] // self.walk_length
        h_reshaped = h.view(num_walks, self.walk_length, self.hidden_size)
        h_reshaped_t = h_reshaped.permute(0, 2, 1)
        M = torch.einsum('ij,ajk->aik', self.components.L, h_reshaped)
        # (w, hidden, walk_len) @ (w, walk_len, hidden) -> (w, hidden, hidden)
        prod = torch.bmm(h_reshaped_t, M)
        trace_per_walk = torch.einsum('wii->w', prod)
        clique_loss = trace_per_walk.mean()

        # trace(H^T * L * H)
        # M = L @ H
        # M = torch.einsum('ij, wjk -> wik', self.components.L, h_reshaped)
        # trace_per_walk = torch.einsum('wld, wld -> w', h_reshaped, M)
        # clique_loss = trace_per_walk.mean()

        print({
            "loss": f"{(recon_loss + sparsity_loss + clique_loss).item():.4f}",
            "recon": f"{recon_loss.item():.2f}",
            "kl": f"{sparsity_loss.item():.4f}",
            "clique": f"{clique_loss.item():.4f}"
        })

        return recon_loss + sparsity_loss + clique_loss

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        epochs = kwargs.get("epochs", None)
        snapshot_i = kwargs.get("snapshot_i", None)
        num_train_snapshots = kwargs.get("num_train_snapshots", None)
        handler = kwargs.get("callback_handler", None)
        if (
            epochs is None
            or snapshot_i is None
            or num_train_snapshots is None
            or handler is None
        ):
            _valid_kwargs = ["epochs", "snapshot_i",
                             "num_train_snapshots", "callback_handler"]
            raise ValueError("`_train_step` missing values for the following kwargs: "
                             f"{', '.join(arg for arg in _valid_kwargs if kwargs.get(arg, None) is None)}")

        ae = self.components.autoencoder
        optimizer = self.components.optimizer

        walk_update = self.components.walk_update
        if walk_update is None:
            raise ValueError(
                "'walk_update' is None, initialize with first snapshot.")

        num_batches = ceil(len(walk_update.walks) / self.batch_size)
        epoch_losses = []

        pbar = tqdm(
            range(epochs), desc=f"Train Snapshot {snapshot_i+1}/{num_train_snapshots}")

        for _ in pbar:
            batch_losses = []
            for i, one_hot_walks_batch in enumerate(walk_update.get_one_hot_walks_batch(self.batch_size)):
                pbar.set_postfix({
                    "loss": f"{epoch_losses[-1].item():.4f}" if len(epoch_losses) > 0 else float("inf"),
                    "batch": f"{i+1}/{num_batches}"
                })

                # print("original shape", one_hot_walks_batch.shape)

                one_hot_walks_batch = one_hot_walks_batch.to(self.device)
                # print("one_hot_walks_batch shape", one_hot_walks_batch.shape)
                one_hot_walks_batch_t = one_hot_walks_batch.T

                # print("input shape", one_hot_walks_batch_t.shape)

                optimizer.zero_grad()
                x_hat, h = ae(one_hot_walks_batch_t, self.corrupt_prob)
                loss = self._compute_loss(one_hot_walks_batch_t, x_hat, h)

                loss.backward()
                optimizer.step()

                batch_losses.append(loss)

            epoch_losses.append(torch.stack(batch_losses).mean())

        return float(torch.stack(epoch_losses).mean())

    def _fit_kmeans(self, edge_embeddings: torch.Tensor) -> None:
        kmeans = KMeans(n_clusters=self.num_clusters)
        self.components.kmeans = kmeans.fit(edge_embeddings)

    def train(self, epochs: int, train_loader: TemporalGraphSnapshotLoader, val_loader: TemporalGraphSnapshotLoader | None = None, callbacks: list[ExperimentCallback] | None = None):
        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        # walk_update = self.components.walk_update

        num_train_snapshots = len(train_loader)
        for i, train_snapshot in enumerate(train_loader):
            self.set_training_mode(True)

            current_graph = train_snapshot.current
            current_edges = current_graph.edges

            cumulative_graph = train_snapshot.cumulative
            if cumulative_graph is None:
                raise ValueError(
                    "Cumulative graph not found in snapshot, set `TemporalSnapshotLoader(..., include_cumulative=True)`.")

            # if walk_update is None:
            #     walk_update = WalkUpdate(
            #         current_edges, train_loader.num_nodes, self.num_walks, self.walk_length, self.reservoir_dim)
            #     self.components.walk_update = walk_update
            # else:
            #     walk_update.update(current_edges)

            self.components.walk_update = WalkUpdate(
                cumulative_graph.edges, train_loader.total_num_nodes, self.num_walks, self.walk_length, self.reservoir_dim)

            ae = Autoencoder(train_loader.total_num_nodes, self.hidden_size)
            optimizer = torch.optim.Adam(
                ae.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

            self.components.autoencoder = ae
            self.components.optimizer = optimizer

            loss = self._train_step(
                train_snapshot, epochs=epochs, snapshot_i=i, num_train_snapshots=num_train_snapshots, callback_handler=handler)

            print("loss", loss)
            state.loss = loss

            cumulative_graph = train_snapshot.cumulative
            if cumulative_graph is None:
                raise ValueError(
                    "Cumulative graph not found in snapshot, set `TemporalSnapshotLoader(..., include_cumulative=True)`.")

            edge_emb = self._get_edge_embeddings(
                cumulative_graph.src, cumulative_graph.tgt)
            self._fit_kmeans(edge_emb)

            if val_loader is not None:
                self.set_training_mode(False)
                val_labels, val_probs = self.run_inference(val_loader)
                val_auc = roc_auc_score(
                    val_labels.cpu().numpy(), val_probs.cpu().numpy())
                state.val_metrics = {'roc_auc': val_auc}
                print("val_auc", val_auc)

            handler.on_train_step_end(state)

        handler.on_train_end(state)

    # def _get_node_embeddings(self):
    #     ae = self.components.autoencoder
    #     ae.eval()
    #     with torch.no_grad():
    #         W_encoder = ae.encoder_layer.weight  # (hidden_size, num_nodes)
    #         b_encoder = ae.encoder_layer.bias
    #         emb = torch.sigmoid(W_encoder.T + b_encoder)
    #         return emb.cpu()  # (num_nodes, hidden_size)

    def _get_node_embeddings(self) -> torch.Tensor:
        ae = self.components.autoencoder
        ae.eval()
        with torch.no_grad():
            I = torch.eye(ae.encoder_layer.in_features).to(self.device)
            embeddings = ae.encode(I)
            return embeddings.cpu()

    def _get_edge_embeddings(self, src: torch.Tensor, tgt: torch.Tensor):
        node_embeddings = self._get_node_embeddings()
        return node_embeddings[src, :] * node_embeddings[tgt, :]

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        kmeans = self.components.kmeans
        if kmeans is None:
            raise ValueError(
                "'self.kmeans' is None, did you fit edge train embeddings?")

        current_graph = snapshot.current
        centroids = kmeans.cluster_centers_
        edge_emb = self._get_edge_embeddings(
            current_graph.src, current_graph.tgt)
        dist_to_centroids = cdist(edge_emb, centroids)  # (num_edges, k)
        min_dist = np.min(dist_to_centroids, axis=1)  # (num_edges,)
        return torch.as_tensor(min_dist)

    def save(self, save_dir: str) -> None:
        return None

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> Self:
        return cls()


if __name__ == "__main__":
    from dgadb.models.baseline.gnn import GNNAD
    from dgadb.preprocessing.add_graph_temporary import inject_anomalies_addgraph_style
    anom_config = {
        "anom_type": "structural",
        "anom_test_ratio": 0.1,
        "anom_val_ratio": 0.1,
    }
    snapshot_config = {
        "strategy": "window",
        "window_size": 1000,
        "include_cumulative": True
    }

    loader = TemporalGraphLoader()
    data = loader.load(
        "bitcoin-alpha", **anom_config, create_if_not_found=True)
    # data = inject_anomalies_addgraph_style(data, **anom_config, seed=1)

    model = Netwalk(
        batch_size=40*3,
        num_clusters=5,
        hidden_size=20,
        ae_weight=1000.0,
        # sparcity_weight=0.1,
        learning_rate=0.001,

    )
    model.setup(data)

    train_loader = TemporalGraphSnapshotLoader(
        data, "window", window_size=1000, split="train", include_cumulative=True)
    val_loader = TemporalGraphSnapshotLoader(
        data, "window", window_size=1000, split="val", include_cumulative=True)
    test_loader = TemporalGraphSnapshotLoader(
        data, "window", window_size=1000, split="test", include_cumulative=True)

    model.train(10, train_loader, val_loader)

    node_embeddings = model._get_node_embeddings()
    print("Embedding Mean:", node_embeddings.mean())
    print("Embedding Std Dev:", node_embeddings.std())

    model.set_training_mode(False)
    val_labels, val_probs = model.run_inference(test_loader)
    val_auc = roc_auc_score(
        val_labels.cpu().numpy(), val_probs.cpu().numpy())
    print("test_auc", val_auc)
