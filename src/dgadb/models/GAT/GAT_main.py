#!/usr/bin/env python3
"""
GAT (Graph Attention Network) Model for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This module implements GAT-based anomaly detection using PyTorch Geometric.
It learns node embeddings through graph attention mechanisms and uses them for edge-level
anomaly detection via a downstream classifier.
"""

import logging
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Tuple, Dict
from torch_geometric.nn import GAT
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from dgadb.storage.graph import Graph
from dgadb.models.common import EdgeDecoder, train_edge_decoder, inference_with_decoder

logger = logging.getLogger(__name__)


class GATModel:
    """GAT-based anomaly detection model.
    
    This class implements Graph Attention Networks for anomaly detection in dynamic graphs.
    It uses GAT to learn node embeddings and then trains a classifier for edge-level anomaly detection.
    
    Attributes:
        device (torch.device): Device to run computations on.
        epoch_evaluation_metric (callable): Function to evaluate model performance.
        hidden_channels (int): Size of hidden layers.
        num_layers (int): Number of GAT layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        classifier_solver (str): Solver for downstream classifier.
        classifier_max_iter (int): Maximum iterations for classifier.
    
    Args:
        device (torch.device): Device to run computations on.
        hyperparams (dict): Dictionary containing model hyperparameters.
        epoch_evaluation_metric (callable): Function that takes (labels, predictions)
            and returns a scalar evaluation metric.
    """
    
    def __init__(
        self,
        device: torch.device,
        hyperparams: dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        
        # Extract GAT hyperparameters with defaults
        self.hidden_channels = hyperparams.get("hidden_channels", 64)
        self.num_layers = hyperparams.get("num_layers", 2)
        self.num_heads = hyperparams.get("num_heads", 8)
        self.dropout = hyperparams.get("dropout", 0.1)
        self.v2 = hyperparams.get("v2", True)  # Use GATv2Conv
        
        # Training parameters
        self.num_epochs = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.01)
        
        # Classifier parameters (deprecated - kept for backward compatibility)
        self.classifier_solver = hyperparams.get("classifier_solver", "lbfgs")
        self.classifier_max_iter = hyperparams.get("classifier_max_iter", 1000)
        
        # Decoder parameters (new configurable parameters)
        self.decoder_epochs = hyperparams.get("decoder_epochs", 100)
        self.decoder_learning_rate = hyperparams.get("decoder_learning_rate", 0.01)
        
        # Initialize placeholders for setup
        self.gat: Optional[GAT] = None
        self.optimizer: Optional[torch.optim.Adam] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.node_features: Optional[torch.Tensor] = None
        self.train_data: Optional[Dict[str, torch.Tensor]] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        self.decoder: Optional[EdgeDecoder] = None
        
        logger.info(f"Initializing GATModel with device={device} and hyperparams={hyperparams}")
    
    def setup(self, graph: Graph) -> None:
        """Set up data processing and initialize the GAT model.
        
        Args:
            graph (Graph): Graph object containing node and edge data.
        """
        logger.info("Setting up GATModel...")
        
        graph.to(self.device)
        
        num_nodes = graph.num_nodes
        
        logger.info(f"Graph has {num_nodes} nodes and {graph.num_edges} edges")
        
        # Create edge index for the entire graph (used for GAT training)
        self.edge_index = graph.e_pairs
        
        # Use edge index as-is from the graph (preprocessing handled by pipeline)
        logger.info(f"Using edge index with {self.edge_index.shape[1]} edges")
        
        # Use node features from the graph object
        self.node_features = graph.n_feat.to(self.device)
        
        # Initialize GAT model
        self.gat = GAT(
            in_channels=self.node_features.shape[1],  # Input feature dimension
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            out_channels=self.hidden_channels,  # Output embedding dimension
            dropout=self.dropout,
            v2=self.v2,
            heads=self.num_heads,
            act='relu'
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.gat.parameters(),
            lr=self.learning_rate
        )
        
        # Prepare data splits for edge-level tasks
        self._prepare_data_splits(graph)
        
        logger.info("GATModel setup completed")
    
    def _prepare_data_splits(self, graph: Graph) -> None:
        """Prepare train/test/val data splits for edge-level anomaly detection."""
        
        # Prepare training data
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
        
        train_count = len(self.train_data['labels']) if self.train_data else 0
        test_count = len(self.test_data['labels']) if self.test_data else 0
        val_count = len(self.val_data['labels']) if self.val_data else 0
        
        logger.info(f"Prepared data splits - Train: {train_count}, Test: {test_count}, Val: {val_count}")
    
    def train(self) -> None:
        """Train the GAT model using unsupervised node embedding learning."""
        self._ensure_setup()
        
        logger.info(f"Starting GAT training for {self.num_epochs} epochs...")
        
        self.gat.train()
        
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass - get node embeddings
            node_embeddings = self.gat(self.node_features, self.edge_index)
            
            # Compute reconstruction loss (link prediction as pretext task)
            loss = self._compute_reconstruction_loss(node_embeddings)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Log progress and validation score
            if (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                log_msg = f"Epoch {epoch + 1}: loss={loss.item():.4f}"
                
                # Add validation score if validation data is available
                if self.val_data is not None:
                    try:
                        val_preds, val_labels, _ = self.inference("val")
                        val_auc = self.epoch_evaluation_metric(val_labels, val_preds)
                        log_msg += f", val_auc={val_auc:.4f}"
                    except Exception as e:
                        logger.warning(f"Could not compute validation score: {e}")
                
                logger.info(log_msg)
        
        logger.info("GAT training completed")
    
    def _compute_reconstruction_loss(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for link prediction pretext task."""
        
        # Get positive edges (existing edges)
        pos_edge_index = self.edge_index
        
        # Sample negative edges
        num_neg_samples = min(pos_edge_index.shape[1], 1000)  # Limit for efficiency
        neg_edge_index = self._sample_negative_edges(num_neg_samples)
        
        # Compute edge scores
        pos_scores = self._compute_edge_scores(node_embeddings, pos_edge_index)
        neg_scores = self._compute_edge_scores(node_embeddings, neg_edge_index)
        
        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        
        return pos_loss + neg_loss
    
    def _sample_negative_edges(self, num_samples: int) -> torch.Tensor:
        """Sample negative edges (non-existing edges) for training using PyTorch Geometric."""
        num_nodes = self.node_features.shape[0]
        
        # Use PyTorch Geometric's optimized negative sampling
        # This is much more efficient than the manual approach
        try:
            neg_edge_index = negative_sampling(
                edge_index=self.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_samples,
                method='sparse'
            )
            return neg_edge_index
        except Exception as e:
            logger.warning(f"PyTorch Geometric negative sampling failed: {e}. Using fallback method.")
            
            # Fallback to a simpler approach if PyG method fails
            # Generate random pairs and filter out existing edges
            max_attempts = num_samples * 5
            neg_edges = []
            existing_edges = set(zip(self.edge_index[0].cpu().numpy(), self.edge_index[1].cpu().numpy()))
            
            attempts = 0
            while len(neg_edges) < num_samples and attempts < max_attempts:
                src = torch.randint(0, num_nodes, (1,)).item()
                tgt = torch.randint(0, num_nodes, (1,)).item()
                
                if src != tgt and (src, tgt) not in existing_edges:
                    neg_edges.append([src, tgt])
                
                attempts += 1
            
            if len(neg_edges) == 0:
                # Final fallback: create some basic edges
                neg_edges = [[0, 1], [1, 2]] if num_nodes > 2 else [[0, 1]]
            
            return torch.tensor(neg_edges, dtype=torch.long).t().to(self.device)
    
    def _compute_edge_scores(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute edge scores using dot product of node embeddings."""
        src_embeddings = node_embeddings[edge_index[0]]
        tgt_embeddings = node_embeddings[edge_index[1]]
        
        # Dot product similarity
        scores = (src_embeddings * tgt_embeddings).sum(dim=1)
        return scores
    
    def inference(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray, float]:
        """Run inference on the specified data split.
        
        Args:
            split: Data split to evaluate ("train", "val", "test")
            
        Returns:
            Tuple of (predictions, labels, inference_time)
        """
        self._ensure_setup()
        start_time = time.time()
        
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
        self.gat.eval()
        with torch.no_grad():
            node_embeddings = self.gat(self.node_features, self.edge_index)
        
        # Use the common decoder for inference
        predictions, labels, inference_time = inference_with_decoder(
            self.decoder, node_embeddings, data['edge_index'], data['labels']
        )
        
        return predictions, labels, inference_time
    
    def _train_decoder(self) -> None:
        """Train downstream decoder for anomaly detection."""
        logger.info("Training downstream decoder...")
        
        if self.train_data is None:
            raise RuntimeError("No training data available for decoder training")
        
        # Get node embeddings
        self.gat.eval()
        with torch.no_grad():
            node_embeddings = self.gat(self.node_features, self.edge_index)
        
        # Initialize decoder
        self.decoder = EdgeDecoder(embedding_dim=self.hidden_channels).to(self.device)
        
        # Train decoder using the common training function with configurable parameters
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
    
    def _get_edge_embeddings(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get edge embeddings by combining node embeddings."""
        src_embeddings = node_embeddings[edge_index[0]]
        tgt_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=1)
        
        return edge_embeddings
    
    def _ensure_setup(self) -> None:
        """Ensure that setup() has been called before training/inference."""
        if any(attr is None for attr in [self.gat, self.optimizer, self.edge_index, self.node_features]):
            raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")