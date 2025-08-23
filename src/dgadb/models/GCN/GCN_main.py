#!/usr/bin/env python3
"""
GCN (Graph Convolutional Network) Model for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This module implements GCN-based anomaly detection using a custom GCN implementation.
It learns node embeddings through graph convolution and uses them for edge-level
anomaly detection via a downstream classifier.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Tuple, Dict

from src.dgadb.storage.graph import Graph
from src.dgadb.models.common import EdgeDecoder, train_edge_decoder, inference_with_decoder
from .gcn_conv import GCNConv

logger = logging.getLogger(__name__)


class GCNCore(nn.Module):
    """
    Core GCN model for learning node embeddings.
    This model stacks multiple GCNConv layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
        remove_self_loops: bool
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            # Single layer maps directly from input to output
            self.convs.append(GCNConv(input_dim, embedding_dim, remove_self_loops=remove_self_loops))
        else:
            # First layer: input -> hidden
            self.convs.append(GCNConv(input_dim, hidden_dim, remove_self_loops=remove_self_loops))
            # Intermediate layers: hidden -> hidden
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim, remove_self_loops=remove_self_loops))
            # Final layer: hidden -> output
            self.convs.append(GCNConv(hidden_dim, embedding_dim, remove_self_loops=remove_self_loops))

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute node embeddings."""
        x = node_features
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply activation and dropout to all but the last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def get_embeddings(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Helper method to get the final node embeddings."""
        return self.forward(node_features, edge_index)


class GCNModel:
    """GCN-based anomaly detection model.
    
    This class implements Graph Convolutional Networks for anomaly detection in dynamic graphs.
    It uses GCN to learn node embeddings and then trains a classifier for edge-level anomaly detection.
    
    Attributes:
        device (torch.device): Device to run computations on.
        epoch_evaluation_metric (callable): Function to evaluate model performance.
        embedding_dim (int): Size of node embeddings.
        hidden_dim (int): Size of hidden layers.
        num_layers (int): Number of GCN layers.
        dropout (float): Dropout probability.
        remove_self_loops (bool): Whether to remove self-loops.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        decoder_epochs (int): Number of epochs for decoder training.
        decoder_learning_rate (float): Learning rate for decoder training.
    
    Args:
        device (torch.device): Device to run computations on.
        hyperparams (dict): Dictionary containing model hyperparameters.
        epoch_evaluation_metric (callable): Function that takes (labels, predictions)
            and returns a scalar evaluation metric.
    """
    
    def __init__(
        self,
        device: torch.device,
        hyperparams: Dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        
        # Extract GCN hyperparameters with defaults
        self.embedding_dim = hyperparams.get("embedding_dim", 128)
        self.hidden_dim = hyperparams.get("hidden_dim", 256)
        self.num_layers = hyperparams.get("num_layers", 2)
        self.dropout = hyperparams.get("dropout", 0.5)
        self.remove_self_loops = hyperparams.get("remove_self_loops", False)
        
        # Training parameters
        self.num_epochs = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.01)
        
        # Decoder parameters (standardized)
        self.decoder_epochs = hyperparams.get("decoder_epochs", 100)
        self.decoder_learning_rate = hyperparams.get("decoder_learning_rate", 0.01)
        
        # Initialize placeholders for setup
        self.model: Optional[GCNCore] = None
        self.optimizer: Optional[torch.optim.Adam] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.node_features: Optional[torch.Tensor] = None
        self.num_nodes: int = 0
        self.train_data: Optional[Dict[str, torch.Tensor]] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        self.decoder: Optional[EdgeDecoder] = None
        
        logger.info(f"Initializing GCNModel with device={device} and hyperparams={hyperparams}")
    
    def setup(self, graph: Graph) -> None:
        """Set up data processing and initialize the GCN model.
        
        Args:
            graph (Graph): Graph object containing node and edge data.
        """
        logger.info("Setting up GCNModel...")
        
        # Move graph to device
        graph.to(self.device)
        
        # Extract graph properties
        self.num_nodes = graph.num_nodes
        self.edge_index = graph.e_pairs
        self.node_features = graph.n_feat  # May be identity matrix
        
        logger.info(f"Graph has {self.num_nodes} nodes and {graph.num_edges} edges")
        
        # Prepare data splits for evaluation
        self._prepare_data_splits(graph)
        
        # Initialize GCN model
        assert self.node_features is not None, "Node features must be set"
        self.model = GCNCore(
            input_dim=self.node_features.shape[1],
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            remove_self_loops=self.remove_self_loops
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        logger.info("GCNModel setup completed")
    
    def _prepare_data_splits(self, graph: Graph) -> None:
        """Prepare train/test/val data splits for edge-level anomaly detection."""
        
        # Standard pattern - prepare training data
        train_mask = graph.e_train_mask
        if train_mask.any():
            self.train_data = {
                'edge_index': graph.e_pairs[:, train_mask],
                'labels': graph.e_label[train_mask].float(),
            }
        
        # Prepare test data
        test_mask = graph.e_test_mask
        if test_mask.any():
            self.test_data = {
                'edge_index': graph.e_pairs[:, test_mask],
                'labels': graph.e_label[test_mask].float(),
            }
        
        # Prepare validation data (if available)
        if hasattr(graph, "e_val_mask"):
            val_mask = graph.e_val_mask
            if val_mask.any():
                self.val_data = {
                    'edge_index': graph.e_pairs[:, val_mask],
                    'labels': graph.e_label[val_mask].float(),
                }
        
        # Log data split sizes
        train_count = len(self.train_data['labels']) if self.train_data else 0
        test_count = len(self.test_data['labels']) if self.test_data else 0
        val_count = len(self.val_data['labels']) if self.val_data else 0
        
        logger.info(f"Prepared data splits - Train: {train_count}, Test: {test_count}, Val: {val_count}")
    
    def train(self) -> None:
        """Train the GCN model using unsupervised link prediction pretext task."""
        self._ensure_setup()
        logger.info(f"Starting GCN training for {self.num_epochs} epochs...")

        # Use the common EdgeDecoder for the pretext training task
        pretext_decoder = EdgeDecoder(self.embedding_dim).to(self.device)
        # Combine GCN and decoder parameters for end-to-end training
        assert self.model is not None, "Model must be initialized"
        assert self.train_data is not None, "Training data must be available"
        combined_params = list(self.model.parameters()) + list(pretext_decoder.parameters())
        optimizer = torch.optim.Adam(combined_params, lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # For efficient negative sampling
        existing_edges = set(tuple(edge) for edge in self.train_data['edge_index'].t().cpu().numpy())
        num_pos_edges = self.train_data['edge_index'].shape[1]

        self.model.train()
        pretext_decoder.train()

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            # --- Negative Sampling ---
            neg_edges: list[list[int]] = []
            while len(neg_edges) < num_pos_edges:
                src = int(torch.randint(0, self.num_nodes, (1,)).item())
                tgt = int(torch.randint(0, self.num_nodes, (1,)).item())
                if src != tgt and (src, tgt) not in existing_edges and (tgt, src) not in existing_edges:
                    neg_edges.append([src, tgt])
            neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(self.device)
            
            # --- End-to-End Forward Pass ---
            assert self.model is not None and self.node_features is not None and self.edge_index is not None
            node_embeddings = self.model(self.node_features, self.edge_index)

            # Get embeddings for positive and negative edges
            pos_src_emb = node_embeddings[self.train_data['edge_index'][0]]
            pos_tgt_emb = node_embeddings[self.train_data['edge_index'][1]]
            neg_src_emb = node_embeddings[neg_edge_index[0]]
            neg_tgt_emb = node_embeddings[neg_edge_index[1]]
            
            # Get predictions from the pretext decoder
            pos_preds = pretext_decoder(pos_src_emb, pos_tgt_emb)
            neg_preds = pretext_decoder(neg_src_emb, neg_tgt_emb)

            # --- Loss Calculation & Backpropagation ---
            preds = torch.cat([pos_preds, neg_preds], dim=0)
            labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=0)
            
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                # Log pretext loss - the true measure of encoder training progress
                log_msg = f"Epoch {epoch + 1}: pretext_loss={loss.item():.4f}"
                logger.info(log_msg)

        logger.info("GCN encoder training completed")
    
    def _train_decoder(self) -> None:
        """Train downstream decoder for anomaly detection."""
        logger.info("Training downstream decoder...")
        
        if self.train_data is None:
            raise RuntimeError("No training data available for decoder training")
        
        # Get node embeddings from trained model
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model.get_embeddings(self.node_features, self.edge_index)
        
        # Initialize decoder
        self.decoder = EdgeDecoder(embedding_dim=self.embedding_dim).to(self.device)
        
        # Train decoder using the common training function
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
        """Run inference on the specified data split.
        
        Args:
            split: Data split to evaluate ("train", "val", "test")
            
        Returns:
            Tuple of (predictions, labels, inference_time)
        """
        self._ensure_setup()
        
        # Select data split
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if data is None:
            raise ValueError(f"No data available for split: {split}")
        
        # Train downstream decoder if not already trained
        if self.decoder is None:
            self._train_decoder()
        
        # Get node embeddings
        assert self.model is not None and self.node_features is not None and self.edge_index is not None
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model.get_embeddings(self.node_features, self.edge_index)
        
        # Use the common decoder for inference
        assert self.decoder is not None, "Decoder must be trained before inference"
        predictions, labels, inference_time = inference_with_decoder(
            self.decoder, node_embeddings, data['edge_index'], data['labels']
        )
        
        return predictions, labels, inference_time
    
    def _ensure_setup(self) -> None:
        """Ensure that setup() has been called before training/inference."""
        if any(attr is None for attr in [self.model, self.optimizer]):
            raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")