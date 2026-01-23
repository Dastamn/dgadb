from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Literal, Type

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GCN, GAT, GraphSAGE
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling

from ..base import BaseADModel, BaseADModelComponents
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshot


_ModelTypeAlias = Type[GCN | GAT | GraphSAGE]
_VALID_MODELS = [GCN, GAT, GraphSAGE]


@dataclass
class GNNADComponents(BaseADModelComponents):
    encoder: GCN | GAT | GraphSAGE
    decoder: InnerProductDecoder
    optimizer: torch.optim.Optimizer
    embedding: torch.nn.Embedding | None = None


def _get_model_class(model_type: Literal["GCN", "GAT", "GraphSAGE"]) -> _ModelTypeAlias:
    for t in _VALID_MODELS:
        if t.__name__.lower() == model_type.lower():
            return t
    raise ValueError(f"Unknown model type: '{model_type}'.")


class GNNAD(BaseADModel[GNNADComponents]):
    def __init__(
        self,
        model_type: Literal["GCN", "GAT", "GraphSAGE"],
        hidden_channels: int = 256,
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
        self.model_class = _get_model_class(model_type)
        self._initial_features = None

    @property
    def initial_features(self):
        if self._initial_features is None:
            raise RuntimeError(
                "Model is not initialized. Call `setup(data)` first.")
        return self._initial_features

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        device = self.device
        if self._components is not None and self._components.embedding is not None:
            # Loaded model with embeddings
            num_nodes = self._components.embedding.num_embeddings
            if data.num_nodes > num_nodes:
                raise ValueError(
                    f"Data has {data.num_nodes} nodes, but the loaded embedding layer only supports {num_nodes}.")

            self._initial_features = torch.arange(num_nodes, device=device)

        elif data.node_attr is not None:
            # Provided data has features
            self._initial_features = data.node_attr.clone().to(device)

        else:
            # New model and no data features
            self._initial_features = torch.arange(
                data.num_nodes, device=device)

        if self._components is not None:
            return

        embedding = None
        if data.node_attr is None:
            embedding = torch.nn.Embedding(
                data.num_nodes, self.hidden_channels)

        encoder = self.model_class(
            in_channels=-1,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            act=self.act
        )

        decoder = InnerProductDecoder()

        params_to_optimize = list(encoder.parameters())
        if embedding is not None:
            params_to_optimize.extend(embedding.parameters())

        optimizer = torch.optim.Adam(
            params=params_to_optimize, lr=self.learning_rate)

        self._components = GNNADComponents(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            embedding=embedding
        ).to(self.device)

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
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

        # data = Data(
        #     x=self.initial_features,
        #     edge_index=cumulative_edge_index,
        #     edge_label_index=current_edge_index
        # )

        # neighbor_loader = LinkNeighborLoader(
        #     data,
        #     num_neighbors=[-1] * self.num_layers,
        #     batch_size=512,
        #     edge_label_index=data.edge_label_index,
        #     neg_sampling_ratio=1.0,
        #     shuffle=False,
        # )

        # for batch in neighbor_loader:
        #     batch = batch.to(device)
        #     return 0

        optimizer.zero_grad()

        x = self.initial_features
        if self.components.embedding is not None:
            x = self.components.embedding(x)

        node_embeddings = encoder(
            x,
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

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        device = self.device
        encoder = self.components.encoder

        self.set_training_mode(False)
        with torch.no_grad():
            cumulative_graph = snapshot.cumulative
            if cumulative_graph is None:
                raise RuntimeError(
                    "Cumulative graph not found in snapshot, set `TemporalSnapshotLoader(..., include_cumulative=True)`.")

            cumulative_edge_index = cumulative_graph.edge_index.to(device)
            cumulative_msg = cumulative_graph.msg.to(device)

            x = self.initial_features
            if self.components.embedding is not None:
                x = self.components.embedding(x)

            node_embeddings = encoder(
                x,
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

        checkpoint = {
            'encoder_state_dict': c.encoder.state_dict(),
            'optimizer_state_dict': c.optimizer.state_dict(),
        }

        config_path = os.path.join(save_dir, "config.json")
        config = {
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "act": self.act,
            "learning_rate": self.learning_rate,
        }

        if c.embedding is not None:
            checkpoint['embedding_state_dict'] = c.embedding.state_dict()
            config["uses_embedding"] = True
            config["num_nodes"] = c.embedding.num_embeddings

        model_path = os.path.join(save_dir, "model.pt")
        torch.save(checkpoint, model_path)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs):
        config_path = os.path.join(load_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at '{config_path}'")

        with open(config_path, 'r') as f:
            config = json.load(f)

        model_type = kwargs.get("model_type", None)
        if model_type is None:
            raise ValueError(
                f"`model_type` not specified, expected any of {[m.__name__ for m in _VALID_MODELS]}")
        model_class = _get_model_class(model_type)

        uses_embedding = config.pop("uses_embedding", False)
        num_nodes = config.pop("num_nodes", None)

        instance = cls(model_type=model_type, device=device, **config)

        model_path = os.path.join(load_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'")

        checkpoint = torch.load(model_path, map_location=device)

        embedding = None
        if uses_embedding:
            if num_nodes is None:
                raise ValueError(
                    "'num_nodes' should not be None when using embeddings.")

            embedding_dim = instance.hidden_channels
            embedding = torch.nn.Embedding(num_nodes, embedding_dim)
            embedding.load_state_dict(checkpoint['embedding_state_dict'])

        encoder = model_class(
            in_channels=-1,
            hidden_channels=instance.hidden_channels,
            num_layers=instance.num_layers,
            act=instance.act
        )
        encoder.load_state_dict(checkpoint['encoder_state_dict'])

        decoder = InnerProductDecoder()

        params_to_optimize = list(encoder.parameters())
        if embedding is not None:
            params_to_optimize.extend(embedding.parameters())

        optimizer = torch.optim.Adam(
            params_to_optimize, lr=instance.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        instance._components = GNNADComponents(
            encoder=encoder, decoder=decoder, optimizer=optimizer, embedding=embedding).to(device)

        return instance
