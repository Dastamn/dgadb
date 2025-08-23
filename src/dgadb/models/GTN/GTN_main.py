#!/usr/bin/env python3
"""
GTN (Graph Transformer Network) Model for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This implementation adapts the Graph Transformer Network for edge-level anomaly detection.
GTN learns new graph structures (meta-paths) from input graphs and uses them to generate
node embeddings for downstream anomaly detection tasks.

Key adaptation: Treats homogeneous graphs as heterogeneous with two edge types:
1. Original graph edges
2. Identity matrix (self-loops)

This allows GTN to learn variable-length meta-paths as described in the original paper.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Tuple, Dict, List

from src.dgadb.storage.graph import Graph
from src.dgadb.models.common import EdgeDecoder, train_edge_decoder, inference_with_decoder
from src.dgadb.models.GCN.gcn_conv import GCNConv

logger = logging.getLogger(__name__)


class GTConv(nn.Module):
    """Graph Transformer Convolution layer for learning meta-path weights.
    
    This implements the simple 1x1 convolution approach from the GTN paper
    to learn combination weights for adjacency matrices.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Simple learnable weights for combining adjacency matrices
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A: List[Tuple[torch.Tensor, torch.Tensor]], num_nodes: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass to compute weighted combinations of adjacency matrices.
        
        Args:
            A: List of (edge_index, edge_weight) tuples for each input adjacency matrix
            num_nodes: Number of nodes
            
        Returns:
            List of (edge_index, edge_weight) tuples for each output channel
        """
        # Apply softmax to get normalized weights for each output channel
        weights = F.softmax(self.weight, dim=1)  # [out_channels, in_channels]
        
        H = []
        for i in range(self.out_channels):
            # Get weights for this output channel
            channel_weights = weights[i]  # [in_channels]
            
            # Combine adjacency matrices using these weights
            combined_edges = []
            combined_weights = []
            
            for j, (edge_index, edge_weight) in enumerate(A):
                # Weight the edges by the learned combination weight
                weighted_edges = channel_weights[j] * edge_weight
                combined_edges.append(edge_index)
                combined_weights.append(weighted_edges)
            
            # Concatenate all weighted edge types
            if combined_edges:
                combined_edge_index = torch.cat(combined_edges, dim=1)
                combined_edge_weight = torch.cat(combined_weights, dim=0)
                H.append((combined_edge_index, combined_edge_weight))
        
        return H


class GTLayer(nn.Module):
    """Graph Transformer Layer for learning meta-paths through matrix multiplication.
    
    This implements the core GTN mechanism: learning to compose adjacency matrices
    through matrix multiplication to generate new meta-path structures.
    """
    
    def __init__(self, num_edge_types: int, num_channels: int, first: bool = True):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.num_channels = num_channels
        self.first = first
        
        if self.first:
            # First layer: two GTConv layers to learn Q1 and Q2 from input edge types
            self.conv1 = GTConv(num_edge_types, num_channels)
            self.conv2 = GTConv(num_edge_types, num_channels)
        else:
            # Non-first layer: two GTConv layers to learn Q1 and Q2 from previous layer output
            self.conv1 = GTConv(num_channels, num_channels)
            self.conv2 = GTConv(num_channels, num_channels)

    def forward(self, A: List[Tuple[torch.Tensor, torch.Tensor]], num_nodes: int,
                H_: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Forward pass for GT layer implementing matrix multiplication for meta-path generation.
        
        Args:
            A: List of (edge_index, edge_weight) tuples for each edge type (first layer only)
            num_nodes: Number of nodes
            H_: Previous layer's output (for non-first layers)
            
        Returns:
            H: List of learned meta-path adjacency matrices via matrix multiplication
            W: Learned combination weights from both conv layers
        """
        if self.first:
            # First layer: learn combinations from input edge types A
            input_matrices = A
        else:
            # Non-first layer: learn combinations from previous layer output H_
            if H_ is None:
                raise ValueError("H_ cannot be None for non-first layers")
            input_matrices = H_
        
        # Apply two GTConv layers to get Q1 and Q2 matrices
        Q1 = self.conv1(input_matrices, num_nodes)  # List of C matrices
        Q2 = self.conv2(input_matrices, num_nodes)  # List of C matrices
        
        # Perform matrix multiplication for each channel: H_i = Q1_i * Q2_i
        H = []
        for i in range(self.num_channels):
            # Get the i-th matrices from Q1 and Q2
            q1_edge_index, q1_edge_weight = Q1[i]
            q2_edge_index, q2_edge_weight = Q2[i]
            
            # Convert to sparse tensors for matrix multiplication
            q1_sparse = torch.sparse_coo_tensor(
                q1_edge_index, q1_edge_weight, (num_nodes, num_nodes)
            ).coalesce()
            
            q2_sparse = torch.sparse_coo_tensor(
                q2_edge_index, q2_edge_weight, (num_nodes, num_nodes)
            ).coalesce()
            
            # Perform sparse matrix multiplication: Q1 * Q2
            # This creates new 2-hop meta-paths
            result_sparse = torch.sparse.mm(q1_sparse, q2_sparse).coalesce()
            
            # Convert back to edge_index, edge_weight format
            result_indices = result_sparse.indices()
            result_values = result_sparse.values()
            
            # Filter out zero weights to keep sparse representation efficient
            nonzero_mask = result_values.abs() > 1e-8
            if nonzero_mask.any():
                filtered_indices = result_indices[:, nonzero_mask]
                filtered_values = result_values[nonzero_mask]
                H.append((filtered_indices, filtered_values))
            else:
                # If all values are zero, create empty adjacency matrix
                empty_indices = torch.empty((2, 0), dtype=torch.long, device=q1_edge_index.device)
                empty_values = torch.empty(0, dtype=torch.float, device=q1_edge_index.device)
                H.append((empty_indices, empty_values))
        
        # Return learned weights from both convolution layers for monitoring
        W1 = self.conv1.weight.data.clone()
        W2 = self.conv2.weight.data.clone()
        W = torch.stack([W1, W2])  # [2, num_channels, input_channels]
        
        return H, W


class GTNCore(nn.Module):
    """
    Core GTN model for learning node embeddings.
    This model generates new graph structures (meta-paths) and then performs
    graph convolution on them to produce final node embeddings.
    """
    
    def __init__(self, num_edge_types: int, num_channels: int, in_dim: int,
                 hidden_dim: int, num_layers: int, num_nodes: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        # GT layers for meta-path generation
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GTLayer(num_edge_types, num_channels, first=True))
            else:
                self.layers.append(GTLayer(num_edge_types, num_channels, first=False))

        # GCN layer for convolution on the learned meta-paths
        self.gcn = GCNConv(in_channels=in_dim, out_channels=hidden_dim)

    def normalization(self, H: List[Tuple[torch.Tensor, torch.Tensor]], num_nodes: int):
        """Normalizes the generated adjacency matrices."""
        norm_H = []
        for i in range(self.num_channels):
            edge_index, edge_weight = H[i]
            
            # Create sparse tensor and compute degree
            adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()
            deg = torch.sparse.sum(adj, dim=1).to_dense()
            
            # Compute degree normalization
            deg_inv = 1.0 / deg
            deg_inv[deg_inv == float('inf')] = 0
            
            # Apply normalization
            row, col = edge_index[0], edge_index[1]
            norm_edge_weight = deg_inv[row] * edge_weight
            
            norm_H.append((edge_index, norm_edge_weight))
        
        return norm_H

    def forward(self, A: List[Tuple[torch.Tensor, torch.Tensor]], X: torch.Tensor, num_nodes: int):
        """
        Forward pass to compute node embeddings.
        Returns a tensor of shape [num_nodes, num_channels * hidden_dim].
        """
        # Meta-path Generation through GTN layers
        H: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        for i in range(self.num_layers):
            if i == 0:
                # First layer: use input adjacency matrices A
                H, W = self.layers[i](A, num_nodes)
            else:
                # Subsequent layers: use previous layer output H as input
                H, W = self.layers[i](A, num_nodes, H_=H)
        
        # Normalize the learned meta-path graphs
        if H is None:
            raise RuntimeError("GTN layers failed to generate meta-paths")
        H_normalized = self.normalization(H, num_nodes)

        # Node Feature Propagation on learned meta-path graphs
        final_embeddings = []
        for i in range(self.num_channels):
            edge_index, edge_weight = H_normalized[i]
            
            # Skip empty graphs (can happen after matrix multiplication)
            if edge_index.shape[1] == 0:
                # Create zero embeddings for this channel
                zero_embedding = torch.zeros(num_nodes, self.gcn.out_channels,
                                           device=X.device, dtype=X.dtype)
                final_embeddings.append(zero_embedding)
                continue
            
            # Perform graph convolution on the i-th learned meta-path graph
            channel_embedding = F.relu(
                self.gcn(X, edge_index=edge_index.detach(), edge_weight=edge_weight)
            )
            final_embeddings.append(channel_embedding)
        
        # Concatenate embeddings from all channels
        X_ = torch.cat(final_embeddings, dim=1)
        
        return X_

    def get_embeddings(self, A: List[Tuple[torch.Tensor, torch.Tensor]], 
                      X: torch.Tensor, num_nodes: int):
        """Get node embeddings for inference."""
        return self.forward(A, X, num_nodes)


class GTNModel:
    """GTN-based anomaly detection model for DGADB."""

    def __init__(
        self,
        device: torch.device,
        hyperparams: Dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        """Initialize GTNModel with device, hyperparameters, and evaluation metric."""
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric

        # Standard and GTN-specific Hyperparameters
        self.embedding_dim = hyperparams.get("embedding_dim", 128)
        self.num_channels = hyperparams.get("num_channels", 2)
        self.num_layers = hyperparams.get("num_layers", 1)
        
        # Ensure embedding_dim is divisible by num_channels for GCN hidden dim
        if self.embedding_dim % self.num_channels != 0:
            raise ValueError("embedding_dim must be divisible by num_channels")
        self.hidden_dim = self.embedding_dim // self.num_channels

        self.num_epochs = hyperparams.get("num_epoch", 50)  # GTN may converge faster
        self.learning_rate = hyperparams.get("learning_rate", 0.005)

        self.decoder_epochs = hyperparams.get("decoder_epochs", 100)
        self.decoder_learning_rate = hyperparams.get("decoder_learning_rate", 0.01)

        # Initialize placeholders
        self.model: Optional[GTNCore] = None
        self.decoder: Optional[EdgeDecoder] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.A: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None  # For storing input adjacency matrices
        
        # Data containers
        self.train_data: Optional[Dict[str, torch.Tensor]] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        
        # Graph properties
        self.num_nodes: Optional[int] = None
        self.node_features: Optional[torch.Tensor] = None

        logger.info(f"Initializing GTNModel with device={device} and hyperparams={hyperparams}")

    def setup(self, graph: Graph) -> None:
        """Set up data processing and initialize the model."""
        logger.info("Setting up GTNModel...")
        graph.to(self.device)
        self.num_nodes = graph.num_nodes
        self.node_features = graph.n_feat

        # CRITICAL: Prepare input for GTNCore
        # A is a list of (edge_index, edge_weight) tuples
        self.A = []
        
        # 1. The original graph edges
        edge_weights = torch.ones(graph.e_pairs.shape[1], device=self.device)
        self.A.append((graph.e_pairs, edge_weights))
        
        # 2. The identity matrix (for self-loops and shorter paths)
        identity_edges = torch.arange(0, self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
        identity_weights = torch.ones(self.num_nodes, device=self.device)
        self.A.append((identity_edges, identity_weights))
        
        logger.info(f"Prepared GTN input with {len(self.A)} edge types.")

        self._prepare_data_splits(graph)

        if self.node_features is None:
            raise RuntimeError("Node features must be set before model initialization")
        
        self.model = GTNCore(
            num_edge_types=len(self.A),
            num_channels=self.num_channels,
            in_dim=self.node_features.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_nodes=self.num_nodes
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info("GTNModel setup completed.")

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
        """Train the GTN encoder using a link prediction pretext task."""
        self._ensure_setup()
        logger.info(f"Starting GTN training for {self.num_epochs} epochs...")

        if self.model is None or self.train_data is None or self.A is None or self.node_features is None or self.num_nodes is None:
            raise RuntimeError("Model components not properly initialized")

        # Create pretext decoder for link prediction
        pretext_decoder = EdgeDecoder(self.embedding_dim).to(self.device)
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
            
            # --- Generate Embeddings using GTNCore ---
            node_embeddings = self.model(self.A, self.node_features, self.num_nodes)

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
                log_msg = f"Epoch {epoch + 1}: pretext_loss={loss.item():.4f}"
                logger.info(log_msg)

        logger.info("GTN encoder training completed")

    def _train_decoder(self) -> None:
        """Train downstream decoder for anomaly detection."""
        logger.info("Training downstream decoder...")
        
        if self.train_data is None or self.model is None or self.A is None or self.node_features is None or self.num_nodes is None:
            raise RuntimeError("Required components not available for decoder training")
        
        # Get node embeddings from trained model
        model = self.model  # Type narrowing
        A = self.A
        node_features = self.node_features
        num_nodes = self.num_nodes
        
        model.eval()
        with torch.no_grad():
            node_embeddings = model.get_embeddings(A, node_features, num_nodes)
        
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
        if self.model is None or self.A is None or self.node_features is None or self.num_nodes is None:
            raise RuntimeError("Model components not properly initialized")
        
        # Type narrowing
        model = self.model
        A = self.A
        node_features = self.node_features
        num_nodes = self.num_nodes
            
        model.eval()
        with torch.no_grad():
            node_embeddings = model.get_embeddings(A, node_features, num_nodes)
        
        # Use the common decoder for inference
        if self.decoder is None:
            raise RuntimeError("Decoder not trained")
        
        decoder = self.decoder  # Type narrowing
        predictions, labels, inference_time = inference_with_decoder(
            decoder, node_embeddings, data['edge_index'], data['labels']
        )
        
        return predictions, labels, inference_time

    def _ensure_setup(self) -> None:
        """Ensure that setup() has been called before training/inference."""
        if any(attr is None for attr in [self.model, self.optimizer]):
            raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")