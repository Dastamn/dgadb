from __future__ import annotations

import os
import torch
import numpy as np
import scipy.sparse as ssp
from dataclasses import dataclass
from typing import Optional, Any, List

# Original StrGNN imports
from dgadb.models.StrGNN.pytorch_DGCNN.main import Classifier, loop_dataset
from dgadb.models.StrGNN.detection.util_functions import (
    generate_node2vec_embeddings,
    dyn_links2subgraphs
)
from dgadb.experiment.callbacks import ExperimentCallback, ExperimentCallbackHandler

from ..base import BaseADModel, BaseADModelComponents, TrainingState
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshot, TemporalGraphSnapshotLoader
import math
from sklearn.metrics import roc_auc_score


@dataclass
class StrGNNComponents(BaseADModelComponents):
    classifier: Classifier
    optimizer: torch.optim.Optimizer


class StrGNNAD(BaseADModel[StrGNNComponents]):
    def __init__(
        self,
        hop: int = 1,
        window_size: int = 5,
        latent_dim: List[int] = [32, 32, 32, 1],
        hidden: int = 128,
        sortpooling_k: int = 30,
        dropout: float = 0.5,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        use_embedding: bool = True,
        device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device)
        self.hop = hop
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.sortpooling_k = sortpooling_k
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_embedding = use_embedding

        # Storage for pre-computed subgraphs
        self._train_graphs = []
        self._test_graphs = []

        self.data_dict = {}

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        print("STRGNN setup started with Negative Sampling...")

        num_nodes = data.num_nodes
        src = data.src.cpu().numpy()
        tgt = data.tgt.cpu().numpy()
        timestamps = data.t.cpu().numpy()

        # We assume all edges in the graph provided are positive (Normal)
        train_mask = data.train_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
        val_mask = data.val_mask.cpu().numpy() if data.val_mask is not None else None

        # 2. Define Fixed Window Size (e.g., 1000 edges per snapshot)
        edges_per_snapshot = 2000
        num_total_edges = len(src)
        # Calculate snapshot ID for every edge: 0, 0, ... (1000 times), 1, 1, ...
        snapshot_ids = np.floor(
            np.arange(num_total_edges) / edges_per_snapshot).astype(int)
        num_snapshots = int(snapshot_ids.max() + 1)

        print(
            f"Created {num_snapshots} snapshots with {edges_per_snapshot} edges each.")

        # 1. Create adjacency matrices for each snapshot
        net = []
        A_train_all = ssp.csr_matrix((num_nodes, num_nodes))
        max_train_snap = snapshot_ids[train_mask].max() if any(
            train_mask) else 0

        for snap_id in range(num_snapshots):
            snap_mask = snapshot_ids == snap_id
            s, t = src[snap_mask], tgt[snap_mask]

            if len(s) > 0:
                A_snap = ssp.csr_matrix(
                    (np.ones_like(s), (s, t)), shape=(num_nodes, num_nodes))
                A_snap = A_snap + A_snap.transpose()
                A_snap.setdiag(0)
                if snap_id <= max_train_snap:
                    A_train_all = A_train_all + A_snap
            else:
                A_snap = ssp.csr_matrix((num_nodes, num_nodes))
            net.append(A_snap)

        # 2. Node2Vec embeddings
        node_information = None
        if self.use_embedding:
            print("Generating Node2Vec embeddings...")
            node_information = generate_node2vec_embeddings(A_train_all, 128)

        # 3. Separate Positive Train Edges
        train_pos_src = src[train_mask]
        train_pos_tgt = tgt[train_mask]
        train_pos_ids = snapshot_ids[train_mask]

        # 4. NEGATIVE SAMPLING (Crucial step)
        print("Sampling negative training links...")
        train_neg_src = []
        train_neg_tgt = []
        train_neg_ids = []

        # For every positive edge at snapshot T, find a pair (u, v) that is NOT connected at T
        for i in range(len(train_pos_ids)):
            snap_id = train_pos_ids[i]
            A_snap = net[snap_id]

            # Keep trying random pairs until we find a non-edge
            while True:
                u = np.random.randint(0, num_nodes)
                v = np.random.randint(0, num_nodes)
                if u != v and A_snap[u, v] == 0:
                    train_neg_src.append(u)
                    train_neg_tgt.append(v)
                    train_neg_ids.append(snap_id)
                    break

        # Convert to numpy for consistency
        train_neg_src = np.array(train_neg_src)
        train_neg_tgt = np.array(train_neg_tgt)
        train_neg_ids = np.array(train_neg_ids)

        # 2. Node2Vec embeddings
        node_information = None
        if self.use_embedding:
            print("Generating Node2Vec embeddings...")
            node_information = generate_node2vec_embeddings(
                A_train_all, 128, True, train_neg=(train_neg_src, train_neg_tgt))

        # 5. Store in data_dict
        self.data_dict["train_pos"] = (train_pos_src, train_pos_tgt)
        self.data_dict["train_neg"] = (train_neg_src, train_neg_tgt)
        self.data_dict["train_pos_id"] = train_pos_ids
        self.data_dict["train_neg_id"] = train_neg_ids

        # For test, we assume test_mask already contains the anomalies/negatives
        # (usually provided by the dataset labels)
        edge_labels = data.edge_labels.cpu().numpy()
        self.data_dict["test_pos"] = (
            src[(test_mask) & (edge_labels == 1)], tgt[(test_mask) & (edge_labels == 1)])
        self.data_dict["test_neg"] = (
            src[(test_mask) & (edge_labels == 0)], tgt[(test_mask) & (edge_labels == 0)])
        self.data_dict["test_pos_id"] = snapshot_ids[(
            test_mask) & (edge_labels == 1)]
        self.data_dict["test_neg_id"] = snapshot_ids[(
            test_mask) & (edge_labels == 0)]

        if data.val_mask is not None and data.val_mask.sum():
            val_src = src[data.val_mask]
            val_tgt = tgt[data.val_mask]
            val_snap_ids = snapshot_ids[data.val_mask]
            val_labels = edge_labels[data.val_mask]

            self.data_dict["val_pos"] = (
                val_src[val_labels == 1], val_tgt[val_labels == 1])
            self.data_dict["val_neg"] = (
                val_src[val_labels == 0], val_tgt[val_labels == 0])
            self.data_dict["val_pos_id"] = val_snap_ids[val_labels == 1]
            self.data_dict["val_neg_id"] = val_snap_ids[val_labels == 0]

            # 6. Extract Subgraphs
            print("Extracting subgraphs with VALIDATION...")
            train_graphs, test_graphs, val_graphs, max_n_label = self.__dyn_links2subgraphs_with_val(
                net,
                self.window_size,
                self.data_dict["train_pos_id"],
                self.data_dict["train_pos"],
                self.data_dict["train_neg_id"],
                self.data_dict["train_neg"],
                self.data_dict["test_pos_id"],
                self.data_dict["test_pos"],
                self.data_dict["test_neg_id"],
                self.data_dict["test_neg"],
                self.data_dict["val_pos_id"],
                self.data_dict["val_pos"],
                self.data_dict["val_neg_id"],
                self.data_dict["val_neg"],
                h=self.hop,
                node_information=node_information,
            )
            self.data_dict.update(
                {"train_graphs": train_graphs, "test_graphs": test_graphs, "val_graphs": val_graphs})

            all_graphs = [g for sublist in (
                train_graphs + val_graphs + test_graphs) for g in sublist]
            num_nodes_list = sorted([g.num_nodes for g in all_graphs])
            k_idx = int(math.ceil(0.6 * len(num_nodes_list))) - 1
            self.sortpooling_k = max(10, num_nodes_list[k_idx])
        else:
            train_graphs, test_graphs, max_n_label = dyn_links2subgraphs(
                net, self.window_size,
                self.data_dict["train_pos_id"], self.data_dict["train_pos"],
                self.data_dict["train_neg_id"], self.data_dict["train_neg"],
                self.data_dict["test_pos_id"], self.data_dict["test_pos"],
                self.data_dict["test_neg_id"], self.data_dict["test_neg"],
                h=self.hop, node_information=node_information
            )
            self.data_dict.update(
                {"train_graphs": train_graphs, "test_graphs": test_graphs})

            # 7. Dynamic K and Model Init (Same as previous fix)
            all_graphs = [g for sublist in (
                train_graphs + test_graphs) for g in sublist]
            num_nodes_list = sorted([g.num_nodes for g in all_graphs])
            k_idx = int(math.ceil(0.6 * len(num_nodes_list))) - 1
            self.sortpooling_k = max(10, num_nodes_list[k_idx])

        print("sortpooling_k", self.sortpooling_k)

        self.feat_dim = max_n_label + 1
        self.attr_dim = node_information.shape[1] if node_information is not None else 0

        self.classifier = Classifier(
            gm="DGCNN", latent_dim=[32, 32, 32, 1], out_dim=0,
            feat_dim=self.feat_dim, attr_dim=self.attr_dim, edge_feat_dim=0,
            sortpooling_k=int(self.sortpooling_k), conv1d_activation="relu",
            hidden=128, num_class=2, dropout=0.5,
            mode="gpu" if "cuda" in str(self.device) else "cpu"
        )

        if "cuda" in str(self.device):
            self.classifier = self.classifier.cuda()

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.learning_rate)

    def __dyn_links2subgraphs_with_val(self, net, window_size, train_pos_id, train_pos, train_neg_id, train_neg,
                                       test_pos_id, test_pos, test_neg_id, test_neg,
                                       val_pos_id, val_pos, val_neg_id, val_neg, **kwargs):

        train_graphs, test_graphs, max_n_label = dyn_links2subgraphs(
            net, window_size, train_pos_id, train_pos, train_neg_id, train_neg,
            test_pos_id, test_pos, test_neg_id, test_neg, **kwargs
        )

        _, val_graphs, _ = dyn_links2subgraphs(
            net, window_size, train_pos_id, train_pos, train_neg_id, train_neg,
            val_pos_id, val_pos, val_neg_id, val_neg, **kwargs
        )

        return train_graphs, test_graphs, val_graphs, max_n_label

    def train(self, epochs: int, train_loader: TemporalGraphSnapshotLoader, val_loader: TemporalGraphSnapshotLoader | None = None, callbacks: List[ExperimentCallback] | None = None):
        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        print(f"Starting STRGNN training for {epochs} epochs...")
        train_graphs = self.data_dict["train_graphs"]
        val_graphs = self.data_dict.get("val_graphs", None)
        train_idxes = list(range(len(train_graphs)))

        for epoch in range(epochs):
            self.classifier.train()
            print(f"EPOCH {epoch}")
            np.random.shuffle(train_idxes)
            self.classifier.train()
            avg_loss, labels, preds = loop_dataset(
                train_graphs, self.classifier, train_idxes, optimizer=self.optimizer, bsize=self.batch_size
            )

            # Compute training score
            train_score = roc_auc_score(labels, preds)
            print(
                f"Epoch {epoch}: train loss={avg_loss[0]:.5f}, train score={train_score:.5f}")

            if val_graphs is not None:
                self.classifier.eval()
                avg_loss, labels, preds = loop_dataset(
                    val_graphs, self.classifier, list(range(len(val_graphs))), bsize=self.batch_size)
                val_score = roc_auc_score(labels, preds)
                print(f"Epoch {epoch}: val score={val_score:.5f}")

    def run_inference(self, loader: TemporalGraphSnapshotLoader) -> tuple[Tensor, Tensor]:
        self.classifier.eval()
        graphs = self.data_dict["test_graphs"]
        avg_loss, labels, preds = loop_dataset(
            graphs, self.classifier, list(range(len(graphs))), bsize=self.batch_size)
        return preds, labels

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        return 0

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        return torch.Tensor(0)

    def save(self, save_dir: str) -> None:
        return None

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> Self:
        pass
