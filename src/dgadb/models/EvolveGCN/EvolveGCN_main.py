#!/usr/bin/env python3
"""
EvolveGCN Model for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This module implements the EvolveGCN-O model. It captures graph dynamics
by evolving the GCN layer weights over time using an LSTM.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Tuple, Dict, List

from src.dgadb.storage.graph import Graph
from src.dgadb.storage.temporal_graph_data import TemporalGraphData
from src.dgadb.models.common import EdgeDecoder, train_edge_decoder, inference_with_decoder
from src.dgadb.models.GCN.gcn_conv import GCNConv
from src.dgadb.preprocessing.snapshotting import assign_snapshots

logger = logging.getLogger(__name__)


class EvolveGCNConv(nn.Module):
    """
    Implements the Evolving Graph Convolution Unit (EGCU-O).
    It uses an LSTMCell to evolve the weights of a GCNConv layer over time.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # The base GCN layer whose weights will be evolved
        self.gcn_conv = GCNConv(in_channels, out_channels, bias=False, **kwargs)

        # LSTMCell evolves the GCN weights. The input is the flattened weight matrix.
        weight_shape = self.gcn_conv.weight.shape
        self.weight_lstm = nn.LSTMCell(in_channels * out_channels, in_channels * out_channels)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                prev_hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs one time-step of the EvolveGCN-O layer.
        """
        # Previous GCN weights are derived from the LSTM hidden state
        prev_gcn_weights_flat = prev_hidden_state[0]

        # Evolve weights using the LSTMCell
        new_h, new_c = self.weight_lstm(prev_gcn_weights_flat, prev_hidden_state)
        new_gcn_weights = new_h.view(self.in_channels, self.out_channels)

        # Set the GCN layer's weight for the current forward pass
        self.gcn_conv.weight = nn.Parameter(new_gcn_weights)

        # Perform graph convolution with the new weights
        new_node_embeddings = self.gcn_conv(node_features, edge_index)

        return new_node_embeddings, (new_h, new_c)


class EvolveGCNCore(nn.Module):
    """
    Core EvolveGCN model that processes a sequence of graph snapshots.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, num_layers: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.convs = nn.ModuleList()
        self.hidden_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

        # Stacking EvolveGCNConv layers
        if num_layers == 1:
            self.convs.append(EvolveGCNConv(input_dim, embedding_dim, **kwargs))
        else:
            self.convs.append(EvolveGCNConv(input_dim, hidden_dim, **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(EvolveGCNConv(hidden_dim, hidden_dim, **kwargs))
            self.convs.append(EvolveGCNConv(hidden_dim, embedding_dim, **kwargs))

    def reset_hidden_states(self, device: torch.device):
        """Initializes the LSTM hidden states for each layer."""
        self.hidden_states = []
        for conv in self.convs:
            lstm_state_size = conv.in_channels * conv.out_channels
            h = torch.zeros(1, lstm_state_size, device=device)
            c = torch.zeros(1, lstm_state_size, device=device)
            self.hidden_states.append((h, c))

    def forward(self, snapshots: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through the sequence of graph snapshots.
        """
        final_node_embeddings = None
        for snapshot_features, snapshot_edge_index in snapshots:
            x = snapshot_features
            next_hidden_states = []

            for i, conv in enumerate(self.convs):
                x, new_hidden_state = conv(x, snapshot_edge_index, self.hidden_states[i])
                next_hidden_states.append(new_hidden_state)

                if i < self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)

            self.hidden_states = next_hidden_states
            final_node_embeddings = x

        if final_node_embeddings is None:
            raise ValueError("Forward pass requires at least one snapshot.")

        return final_node_embeddings

    def get_embeddings(self, snapshots: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.forward(snapshots)


class EvolveGCNModel:
    """EvolveGCN-based anomaly detection model for the DGADB framework."""

    def __init__(self, device: torch.device, hyperparams: Dict[str, Any], epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float]) -> None:
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric

        # Model hyperparameters
        self.embedding_dim = hyperparams.get("embedding_dim", 128)
        self.hidden_dim = hyperparams.get("hidden_dim", 256)
        self.num_layers = hyperparams.get("num_layers", 2)
        self.dropout = hyperparams.get("dropout", 0.5)
        self.remove_self_loops = hyperparams.get("remove_self_loops", False)
        self.variant = hyperparams.get("variant", "O")
        if self.variant != 'O':
            raise NotImplementedError("Only EvolveGCN-O variant is implemented.")

        # Training and decoder parameters
        self.num_epochs = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.01)
        self.decoder_epochs = hyperparams.get("decoder_epochs", 100)
        self.decoder_learning_rate = hyperparams.get("decoder_learning_rate", 0.01)

        # Placeholders
        self.model: Optional[EvolveGCNCore] = None
        self.optimizer: Optional[torch.optim.Adam] = None
        self.num_nodes: int = 0
        self.snapshots: List[Tuple[Optional[torch.Tensor], torch.Tensor]] = []
        self.final_node_features: Optional[torch.Tensor] = None
        self.train_data: Optional[Dict[str, torch.Tensor]] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        self.decoder: Optional[EdgeDecoder] = None
        self.input_projection: Optional[nn.Linear] = None

        logger.info(f"Initializing EvolveGCNModel with device={device} and hyperparams={hyperparams}")

    def setup(self, temporal_graph: TemporalGraphData, graph: Graph) -> None:
        """
        Set up the model using temporal data and a final graph for evaluation splits.
        """
        logger.info("Setting up EvolveGCNModel...")
        graph.to(self.device)
        self.num_nodes = graph.num_nodes
        self.final_node_features = graph.n_feat

        # Create snapshots from the temporal graph object
        assert self.final_node_features is not None, "Node features must be available"
        
        # Convert temporal graph data to DataFrame format for snapshotting
        import polars as pl
        
        # Create edges DataFrame from temporal graph
        edges_df = pl.DataFrame({
            "src": temporal_graph.src.cpu().numpy(),
            "tgt": temporal_graph.tgt.cpu().numpy(),
            "timestamp": temporal_graph.t.cpu().numpy(),
            "train_mask": temporal_graph.train_mask.cpu().numpy(),
            "test_mask": temporal_graph.test_mask.cpu().numpy(),
        })
        
        # Add val_mask if it exists
        if hasattr(temporal_graph, 'val_mask') and temporal_graph.val_mask is not None:
            edges_df = edges_df.with_columns(pl.Series("val_mask", temporal_graph.val_mask.cpu().numpy()))
        
        # Use the existing snapshotting functionality
        dfs = {"edges": edges_df}
        snapshot_size = max(1000, len(edges_df) // 5)  # Create ~5 snapshots with minimum 1000 edges each
        dfs_with_snapshots = assign_snapshots(dfs, snapshot_size, temporal_snapshots=True)
        
        # Extract snapshots and create graph snapshots
        edges_with_snapshots = dfs_with_snapshots["edges"]
        unique_snapshot_ids = edges_with_snapshots["snapshot_id"].unique().sort()
        
        # Limit to maximum 5 snapshots to avoid memory issues
        max_snapshots = min(5, len(unique_snapshot_ids))
        selected_snapshot_ids = unique_snapshot_ids[:max_snapshots]
        
        for snapshot_id in selected_snapshot_ids:
            snapshot_edges = edges_with_snapshots.filter(pl.col("snapshot_id") == snapshot_id)
            if len(snapshot_edges) > 0:
                # Keep tensors on CPU initially to save GPU memory
                snapshot_src = torch.tensor(snapshot_edges["src"].to_numpy(), dtype=torch.long)
                snapshot_tgt = torch.tensor(snapshot_edges["tgt"].to_numpy(), dtype=torch.long)
                edge_index_t = torch.stack([snapshot_src, snapshot_tgt], dim=0)
                # Store CPU tensors, will move to GPU during forward pass
                self.snapshots.append((None, edge_index_t))  # None for node features, will be set during forward
        
        logger.info(f"Created {len(self.snapshots)} temporal snapshots using the preprocessing snapshotting functionality.")

        # Prepare evaluation data splits from the final aggregated graph
        self._prepare_data_splits(graph)

        # Use a smaller input dimension to avoid memory issues with large identity matrices
        actual_input_dim = min(128, self.final_node_features.shape[1])
        
        # Add a projection layer if needed to reduce dimensionality
        if self.final_node_features.shape[1] > actual_input_dim:
            self.input_projection = nn.Linear(self.final_node_features.shape[1], actual_input_dim).to(self.device)
            logger.info(f"Added input projection from {self.final_node_features.shape[1]} to {actual_input_dim}")
        else:
            self.input_projection = None

        self.model = EvolveGCNCore(
            input_dim=actual_input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            remove_self_loops=self.remove_self_loops
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info("EvolveGCNModel setup completed.")

    def _prepare_data_splits(self, graph: Graph) -> None:
        """Prepare train/test/val data splits from the final graph."""
        train_mask = graph.e_train_mask
        if train_mask.any():
            self.train_data = {'edge_index': graph.e_pairs[:, train_mask], 'labels': graph.e_label[train_mask].float()}
        
        test_mask = graph.e_test_mask
        if test_mask.any():
            self.test_data = {'edge_index': graph.e_pairs[:, test_mask], 'labels': graph.e_label[test_mask].float()}
        
        if hasattr(graph, "e_val_mask") and graph.e_val_mask.any():
            val_mask = graph.e_val_mask
            self.val_data = {'edge_index': graph.e_pairs[:, val_mask], 'labels': graph.e_label[val_mask].float()}
        
        logger.info(f"Prepared data splits - Train: {len(self.train_data['labels']) if self.train_data else 0}, Test: {len(self.test_data['labels']) if self.test_data else 0}, Val: {len(self.val_data['labels']) if self.val_data else 0}")

    def train(self) -> None:
        """Train the model using a link prediction pretext task."""
        self._ensure_setup()
        assert self.model is not None, "Model must be initialized"
        logger.info(f"Starting EvolveGCN training for {self.num_epochs} epochs...")

        pretext_decoder = EdgeDecoder(self.embedding_dim).to(self.device)
        combined_params = list(self.model.parameters()) + list(pretext_decoder.parameters())
        optimizer = torch.optim.Adam(combined_params, lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        assert self.train_data is not None, "Training data must be available"
        existing_edges = set(tuple(edge) for edge in self.train_data['edge_index'].t().cpu().numpy())
        num_pos_edges = self.train_data['edge_index'].shape[1]

        self.model.train()
        pretext_decoder.train()

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            self.model.reset_hidden_states(self.device)

            # Prepare snapshots with node features for this forward pass
            assert self.final_node_features is not None, "Node features must be available"
            snapshots_with_features = []
            for _, edge_index in self.snapshots:
                node_features_t = self.final_node_features.clone().detach()
                # Apply input projection if needed
                if self.input_projection is not None:
                    node_features_t = self.input_projection(node_features_t)
                edge_index_gpu = edge_index.to(self.device)
                snapshots_with_features.append((node_features_t, edge_index_gpu))

            node_embeddings = self.model(snapshots_with_features)

            # Negative Sampling
            neg_edges: List[List[int]] = []
            while len(neg_edges) < num_pos_edges:
                src, tgt = torch.randint(0, self.num_nodes, (2,)).tolist()
                if src != tgt and (src, tgt) not in existing_edges and (tgt, src) not in existing_edges:
                    neg_edges.append([src, tgt])
            neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(self.device)

            # Predictions and Loss
            pos_src_emb = node_embeddings[self.train_data['edge_index'][0]]
            pos_tgt_emb = node_embeddings[self.train_data['edge_index'][1]]
            neg_src_emb = node_embeddings[neg_edge_index[0]]
            neg_tgt_emb = node_embeddings[neg_edge_index[1]]

            pos_preds = pretext_decoder(pos_src_emb, pos_tgt_emb)
            neg_preds = pretext_decoder(neg_src_emb, neg_tgt_emb)

            preds = torch.cat([pos_preds, neg_preds], dim=0)
            labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=0)

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                logger.info(f"Epoch {epoch + 1}: pretext_loss={loss.item():.4f}")

        logger.info("EvolveGCN encoder training completed")

    def _train_decoder(self) -> None:
        """Train the downstream anomaly detection decoder."""
        logger.info("Training downstream decoder...")
        if self.train_data is None:
            raise RuntimeError("No training data available for decoder training")

        assert self.model is not None, "Model must be initialized"
        self.model.eval()
        with torch.no_grad():
            self.model.reset_hidden_states(self.device)
            # Prepare snapshots with node features for inference
            assert self.final_node_features is not None, "Node features must be available"
            snapshots_with_features = []
            for _, edge_index in self.snapshots:
                node_features_t = self.final_node_features.clone().detach()
                # Apply input projection if needed
                if self.input_projection is not None:
                    node_features_t = self.input_projection(node_features_t)
                edge_index_gpu = edge_index.to(self.device)
                snapshots_with_features.append((node_features_t, edge_index_gpu))
            
            node_embeddings = self.model.get_embeddings(snapshots_with_features)

        self.decoder = EdgeDecoder(embedding_dim=self.embedding_dim).to(self.device)
        train_edge_decoder(
            decoder=self.decoder,
            node_embeddings=node_embeddings,
            train_edge_index=self.train_data['edge_index'],
            train_labels=self.train_data['labels'],
            num_epochs=self.decoder_epochs,
            learning_rate=self.decoder_learning_rate,
            device=self.device
        )
        logger.info("Decoder training completed")

    def inference(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray, float]:
        """Run inference on the specified data split."""
        self._ensure_setup()
        data = {"train": self.train_data, "val": self.val_data, "test": self.test_data}.get(split)
        if data is None:
            raise ValueError(f"Unknown or unavailable split: {split}")

        if self.decoder is None:
            self._train_decoder()

        assert self.model is not None, "Model must be initialized"
        self.model.eval()
        with torch.no_grad():
            self.model.reset_hidden_states(self.device)
            # Prepare snapshots with node features for inference
            assert self.final_node_features is not None, "Node features must be available"
            snapshots_with_features = []
            for _, edge_index in self.snapshots:
                node_features_t = self.final_node_features.clone().detach()
                # Apply input projection if needed
                if self.input_projection is not None:
                    node_features_t = self.input_projection(node_features_t)
                edge_index_gpu = edge_index.to(self.device)
                snapshots_with_features.append((node_features_t, edge_index_gpu))
            
            node_embeddings = self.model.get_embeddings(snapshots_with_features)
        
        assert self.decoder is not None
        predictions, labels, inference_time = inference_with_decoder(
            self.decoder, node_embeddings, data['edge_index'], data['labels']
        )
        return predictions, labels, inference_time

    def _ensure_setup(self) -> None:
        if any(attr is None for attr in [self.model, self.optimizer]):
            raise RuntimeError("Model not properly initialized. Call setup() first.")