from __future__ import annotations

import os
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling

from ..base import BaseADModel, BaseADModelComponents
from src.dgadb.storage import TemporalGraphSnapshotLoader
from src.dgadb.storage.temporal_graph import TemporalGraph
from src.dgadb.storage.temporal_snapshot import TemporalGraphSnapshot, TemporalGraphSnapshotLoader


@dataclass
class GCNADComponents(BaseADModelComponents):
    encoder: GCN
    decoder: InnerProductDecoder
    optimizer: torch.optim.Optimizer


class GCNAD(BaseADModel[GCNADComponents]):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        act: str = "relu",
        learning_rate: float = 1e-3,
        device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.act = act
        self.learning_rate = learning_rate

    @property
    def initial_features(self):
        if self._initial_features is None:
            raise RuntimeError(
                "Model is not initialized. Call `setup(data)` first.")
        return self._initial_features

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        device = self.device
        self._initial_features = (
            data.node_attr.clone()
            if data.node_attr is not None
            else torch.eye(data.num_nodes, dtype=torch.float32, device=device)
        ).to(device)

        if self._components is not None:
            return

        encoder = GCN(
            in_channels=-1,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            act=self.act
        )
        decoder = InnerProductDecoder()
        optimizer = torch.optim.Adam(
            params=encoder.parameters(), lr=self.learning_rate
        )

        self._components = GCNADComponents(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer
        ).to(device)

    def _train_step(self, snapshot: TemporalGraphSnapshot) -> float:
        device = self.device
        encoder = self.components.encoder
        decoder = self.components.decoder
        optimizer = self.components.optimizer

        current_graph = snapshot.current
        cumulative_graph = snapshot.cumulative
        if cumulative_graph is None:
            raise RuntimeError(
                "Cumulative graph not found in snapshot, set `TemporalSnapshotLoader(..., include_cumulative=True)`.")

        current_edge_index = current_graph.edge_index.to(device)
        cumulative_msg = cumulative_graph.msg.to(device)
        cumulative_edge_index = cumulative_graph.edge_index.to(device)

        optimizer.zero_grad()

        node_embeddings = encoder(
            self.initial_features,
            edge_index=cumulative_edge_index,
            edge_attr=cumulative_msg
        )

        neg_edge_index = negative_sampling(
            edge_index=cumulative_edge_index,
            num_nodes=cumulative_graph.num_nodes,
            num_neg_samples=current_edge_index.shape[1]
        ).to(device)

        pos_probs = decoder(
            node_embeddings, current_edge_index, sigmoid=True)
        neg_probs = decoder(
            node_embeddings, neg_edge_index, sigmoid=True)
        predictions = torch.cat([pos_probs, neg_probs])

        pos_labels = torch.ones_like(pos_probs, device=device)
        neg_labels = torch.zeros_like(neg_probs, device=device)
        labels = torch.cat([pos_labels, neg_labels])

        loss = F.binary_cross_entropy(predictions, labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self, snapshot: TemporalGraphSnapshot) -> torch.Tensor:
        device = self.device
        encoder = self.components.encoder.to(device)

        self.set_training_mode(False)
        with torch.no_grad():
            cumulative_graph = snapshot.cumulative
            if cumulative_graph is None:
                raise RuntimeError(
                    "Cumulative graph not found in snapshot, set `TemporalSnapshotLoader(..., include_cumulative=True)`.")

            cumulative_edge_index = cumulative_graph.edge_index.to(device)
            cumulative_msg = cumulative_graph.msg.to(device)

            node_embeddings = encoder(
                self.initial_features,
                edge_index=cumulative_edge_index,
                edge_attr=cumulative_msg
            )

            current_graph = snapshot.current
            current_edge_index = current_graph.edge_index.to(device)

            scores = self.components.decoder(
                node_embeddings, current_edge_index, sigmoid=True)
            anomaly_scores = 1.0 - scores

            return anomaly_scores.detach()

    def save(self, save_dir: str) -> None:
        c = self.components
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "model.pt")
        torch.save({
            'encoder_state_dict': c.encoder.state_dict(),
            'optimizer_state_dict': c.optimizer.state_dict()
        }, model_path)

        config_path = os.path.join(save_dir, "config.json")
        config = {
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'act': self.act,
            'learning_rate': self.learning_rate,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu"):
        config_path = os.path.join(load_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at '{config_path}'")
        with open(config_path, 'r') as f:
            config = json.load(f)

        instance = cls(device=device, **config)

        model_path = os.path.join(load_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'")

        checkpoint = torch.load(model_path, map_location=device)
        encoder = GCN(
            in_channels=-1,
            hidden_channels=instance.hidden_channels,
            num_layers=instance.num_layers,
            act=instance.act
        )
        decoder = InnerProductDecoder()
        optimizer = torch.optim.Adam(
            encoder.parameters(), lr=instance.learning_rate)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        instance._components = GCNADComponents(
            encoder=encoder, decoder=decoder, optimizer=optimizer).to(device)

        return instance


if __name__ == "__main__":
    from src.dgadb.storage import TemporalGraphLoader
    model = GCNAD()
    loader = TemporalGraphLoader()
    tg = loader.load("bitcoin-alpha", anom_type="structural", anom_train_ratio=.0,
                     anom_test_ratio=.1, anom_val_ratio=.1)
    train_loader = TemporalGraphSnapshotLoader(
        tg, split="train", window_size=1000, include_cumulative=True)

    val_loader = TemporalGraphSnapshotLoader(
        tg, split="val", window_size=1000, include_cumulative=True)

    model.setup(tg)
    model.train(50, train_loader, val_loader)
