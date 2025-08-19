#!/usr/bin/env python3
"""
NetWalk Model for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This module implements NetWalk-based anomaly detection using PyTorch.
It learns node embeddings through clique embedding with deep autoencoder and uses them for edge-level
anomaly detection via streaming k-means clustering.

Based on the paper:
"NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks"
by Yu et al., KDD 2018
"""

import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Callable, Optional, Tuple, Dict, List
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
from collections import Counter
import random

from src.dgadb.storage.graph import Graph

logger = logging.getLogger(__name__)


class CliqueAutoencoder(nn.Module):
    """Deep autoencoder for clique embedding."""
    
    def __init__(self, input_dim: int, embedding_dim: int, walk_length: int):
        super(CliqueAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        # Create Laplacian matrix for clique loss
        phi = np.ones((walk_length, walk_length)) - np.eye(walk_length)
        L = np.diag(np.sum(phi, axis=1)) - phi
        self.register_buffer('L', torch.tensor(L, dtype=torch.float32))
    
    def _init_weights(self):
        """Initialize weights with uniform unit scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Uniform unit scaling initialization
                fan_in = module.in_features
                limit = np.sqrt(1.0 / fan_in)
                nn.init.uniform_(module.weight, -limit, limit)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, corrupt_prob: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape [num_nodes, batch_size]
            corrupt_prob: Probability of corruption for denoising
            
        Returns:
            Tuple of (reconstruction, encoding)
        """
        # Add corruption for denoising autoencoder
        if corrupt_prob > 0.0 and self.training:
            noise = torch.rand_like(x) * 0.1
            corrupted_x = x * (1 - corrupt_prob) + noise * corrupt_prob
        else:
            corrupted_x = x
        
        # Encode
        encoding = self.encoder(corrupted_x.T)  # Transpose to [batch_size, num_nodes]
        
        # Decode
        reconstruction = self.decoder(encoding)
        
        return reconstruction.T, encoding.T  # Transpose back to [num_nodes, batch_size]


class NetWalkModel:
    """NetWalk-based anomaly detection model.
    
    This class implements NetWalk for anomaly detection in dynamic graphs.
    It uses clique embedding with deep autoencoder to learn node embeddings and then 
    uses streaming k-means clustering for edge-level anomaly detection.
    
    Attributes:
        device (torch.device): Device to run computations on.
        epoch_evaluation_metric (callable): Function to evaluate model performance.
        embedding_dim (int): Size of node embeddings.
        walk_length (int): Length of random walks.
        walks_per_node (int): Number of walks per node.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        k_clusters (int): Number of clusters for anomaly detection.
    """
    
    def __init__(
        self,
        device: torch.device,
        hyperparams: Dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        
        # NetWalk hyperparameters
        self.embedding_dim = hyperparams.get("embedding_dim", 64)
        self.walk_length = hyperparams.get("walk_length", 3)
        self.walks_per_node = hyperparams.get("walks_per_node", 5)
        self.reservoir_dim = hyperparams.get("reservoir_dim", 10)
        
        # Training parameters
        self.num_epochs = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.1)
        self.batch_size = hyperparams.get("batch_size", 20)
        
        # Autoencoder parameters
        self.gamma = hyperparams.get("gamma", 340.0)  # autoencoder weight
        self.lamb = hyperparams.get("lamb", 0.0017)   # weight decay
        self.beta = hyperparams.get("beta", 1.0)      # sparsity weight
        self.rho = hyperparams.get("rho", 0.5)        # sparsity ratio
        
        # Clustering parameters
        self.k_clusters = hyperparams.get("k_clusters", 5)
        self.alpha = hyperparams.get("alpha", 0.5)    # decay factor for streaming k-means
        self.incremental_epochs = hyperparams.get("incremental_epochs", 1)
        
        # Initialize placeholders
        self.num_nodes: Optional[int] = None
        self.node_mapping: Optional[Dict[int, int]] = None
        self.reverse_node_mapping: Optional[Dict[int, int]] = None
        self.reservoir: Optional[Dict[int, np.ndarray]] = None
        self.degree: Optional[Dict[int, int]] = None
        
        # PyTorch model and optimizer
        self.model: Optional[CliqueAutoencoder] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.model_built = False
        self.trained = False
        
        # Data containers
        self.train_data: Optional[Dict[str, torch.Tensor]] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        
        # Clustering state
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_counts: Optional[np.ndarray] = None
        
        logger.info(f"Initializing NetWalkModel with device={device} and hyperparams={hyperparams}")
    
    def setup(self, graph: Graph) -> None:
        """Set up data processing and initialize the NetWalk model.
        
        Args:
            graph (Graph): Graph object containing node and edge data.
        """
        logger.info("Setting up NetWalkModel...")
        
        # Move graph to device
        graph.to(self.device)
        
        self.num_nodes = graph.num_nodes
        
        logger.info(f"Graph has {self.num_nodes} nodes and {graph.num_edges} edges")
        
        # Create node mapping for consistency
        unique_nodes = torch.unique(graph.e_pairs.flatten()).cpu().numpy()
        self.node_mapping = {int(node): idx for idx, node in enumerate(unique_nodes)}
        self.reverse_node_mapping = {idx: int(node) for node, idx in self.node_mapping.items()}
        
        # Initialize reservoir sampling
        self._initialize_reservoir(graph)
        
        # Prepare data splits
        self._prepare_data_splits(graph)
        
        # Build PyTorch model
        self._build_model()
        
        logger.info("NetWalkModel setup completed")
    
    def _initialize_reservoir(self, graph: Graph) -> None:
        """Initialize reservoir sampling for dynamic updates."""
        logger.info("Initializing reservoir sampling...")
        
        self.reservoir = {}
        self.degree = {}
        
        # Convert edge pairs to adjacency list
        edge_pairs = graph.e_pairs.cpu().numpy()
        adjacency: Dict[int, List[int]] = {}
        
        for i in range(edge_pairs.shape[1]):
            src, tgt = edge_pairs[0, i], edge_pairs[1, i]
            if src not in adjacency:
                adjacency[src] = []
            if tgt not in adjacency:
                adjacency[tgt] = []
            adjacency[src].append(tgt)
            adjacency[tgt].append(src)  # Assume undirected
        
        # Initialize reservoir for each node
        if self.num_nodes is not None:
            for node_id in range(self.num_nodes):
                if node_id in adjacency:
                    neighbors = list(set(adjacency[node_id]))  # Remove duplicates
                    self.degree[node_id] = len(neighbors)
                    
                    if len(neighbors) >= self.reservoir_dim:
                        # Sample reservoir_dim neighbors
                        np.random.seed(24)
                        indices = np.random.choice(len(neighbors), self.reservoir_dim, replace=True)
                        self.reservoir[node_id] = np.array([neighbors[idx] for idx in indices])
                    else:
                        # Pad with repetitions if not enough neighbors
                        reservoir = neighbors * (self.reservoir_dim // len(neighbors) + 1)
                        self.reservoir[node_id] = np.array(reservoir[:self.reservoir_dim])
                else:
                    self.reservoir[node_id] = np.array([0] * self.reservoir_dim)
                    self.degree[node_id] = 0
    
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
    
    def _build_model(self) -> None:
        """Build the PyTorch clique embedding model."""
        logger.info("Building PyTorch clique embedding model...")
        
        if self.num_nodes is None:
            raise ValueError("num_nodes is not initialized")
        
        # Create autoencoder model
        self.model = CliqueAutoencoder(
            input_dim=self.num_nodes,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        
        self.model_built = True
        logger.info("PyTorch model built successfully")
    
    def _compute_losses(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                       encoding: torch.Tensor, corrupt_prob: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all loss components."""
        
        # Autoencoder reconstruction loss
        ae_loss = torch.mean((reconstruction - x) ** 2)
        
        # Sparsity constraint (KL divergence)
        rhohats = torch.mean(encoding, dim=1)  # Average activation per hidden unit
        kl_loss = torch.mean(
            self.rho * torch.log(self.rho / (rhohats + 1e-8)) +
            (1 - self.rho) * torch.log((1 - self.rho) / (1 - rhohats + 1e-8))
        )
        
        # Clique loss
        # Reshape encoding for clique computation
        batch_size = encoding.shape[1]
        trans_code = encoding.T  # [batch_size, embedding_dim]
        trans_code = trans_code.view(-1, self.walk_length, self.embedding_dim)  # [batch_size//walk_length, walk_length, embedding_dim]
        
        # Compute clique loss using Einstein summation
        # left = t_trans_code @ L
        assert self.model is not None, "Model must be initialized"
        left = torch.einsum('aij,jk->aik', trans_code.transpose(1, 2), self.model.L)
        # mul = left @ trans_code
        mul = torch.einsum('aij,ajk->aik', left, trans_code)
        # trace of mul
        trace_mul = torch.diagonal(mul, dim1=-2, dim2=-1).sum(-1)
        clique_loss = torch.mean(trace_mul)
        
        # Weight decay
        weight_decay_tensor = torch.tensor(0.0, device=self.device)
        assert self.model is not None, "Model must be initialized"
        for param in self.model.parameters():
            weight_decay_tensor += (self.lamb / 2.0) * torch.mean(param ** 2)
        
        # Total loss
        total_loss = clique_loss + self.gamma * ae_loss + self.beta * kl_loss + weight_decay_tensor
        
        return total_loss, clique_loss, ae_loss, kl_loss, weight_decay_tensor
    
    def _generate_walks(self) -> np.ndarray:
        """Generate random walks using reservoir sampling."""
        walks = []
        
        if self.num_nodes is not None:
            for node_id in range(self.num_nodes):
                for _ in range(self.walks_per_node):
                    walk = [node_id]
                    current = node_id
                    
                    for _ in range(self.walk_length - 1):
                        if self.reservoir is not None and current in self.reservoir and len(self.reservoir[current]) > 0:
                            # Filter out None values
                            valid_neighbors = [n for n in self.reservoir[current] if n is not None]
                            if valid_neighbors:
                                next_node = np.random.choice(valid_neighbors)
                                walk.append(next_node)
                                current = next_node
                            else:
                                break
                        else:
                            break
                    
                    # Pad walk if too short
                    while len(walk) < self.walk_length:
                        walk.append(walk[-1])
                    
                    walks.append(walk[:self.walk_length])
        
        return np.array(walks)
    
    def _walks_to_onehot(self, walks: np.ndarray) -> torch.Tensor:
        """Convert walks to one-hot encoded format."""
        walk_mat = walks.flatten()
        rows = walk_mat
        cols = np.arange(len(rows))
        data = np.ones(len(rows))
        
        # Create sparse matrix and convert to dense
        from scipy.sparse import coo_matrix
        coo = coo_matrix((data, (rows, cols)), shape=(self.num_nodes, len(rows)))
        onehot_tensor = torch.tensor(coo.toarray(), dtype=torch.float32).to(self.device)
        return onehot_tensor
    
    def train(self) -> None:
        """Train the NetWalk model using clique embedding on the initial graph."""
        if not self.model_built:
            raise RuntimeError("Model not built. Call setup() first.")
        
        logger.info(f"Starting NetWalk initial training for {self.num_epochs} epochs...")
        
        # Generate walks
        walks = self._generate_walks()
        onehot_walks = self._walks_to_onehot(walks)
        
        # Training loop
        assert self.model is not None, "Model must be initialized"
        assert self.optimizer is not None, "Optimizer must be initialized"
        self.model.train()
        for epoch in range(self.num_epochs):
            # Batch the data
            batch_size = self.batch_size * self.walk_length
            num_batches = onehot_walks.shape[1] // batch_size
            
            total_loss = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, onehot_walks.shape[1])
                batch_data = onehot_walks[:, start_idx:end_idx]
                
                # Forward pass
                reconstruction, encoding = self.model(batch_data, corrupt_prob=0.0)
                
                # Compute losses
                loss, clique_loss, ae_loss, kl_loss, weight_decay = self._compute_losses(
                    batch_data, reconstruction, encoding
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Log progress and validation score
            if (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                avg_loss = total_loss / max(num_batches, 1)
                log_msg = f"Epoch {epoch + 1}: loss={avg_loss:.4f}"
                
                # Add validation score if validation data is available
                if self.val_data is not None:
                    try:
                        val_preds, val_labels, _ = self._batch_inference(self.val_data)
                        val_auc = self.epoch_evaluation_metric(val_labels, val_preds)
                        log_msg += f", val_auc={val_auc:.4f}"
                    except Exception as e:
                        logger.warning(f"Could not compute validation score: {e}")
                
                logger.info(log_msg)
        
        self.trained = True
        logger.info("NetWalk training completed")

    def _update_reservoirs_and_degrees(self, new_edges: np.ndarray) -> None:
        """Updates node degrees and reservoirs for incoming edges."""
        for i in range(new_edges.shape[1]):
            u, v = new_edges[0, i], new_edges[1, i]

            # Handle new nodes that were not in the initial graph
            for node in [u, v]:
                if self.degree is not None and node not in self.degree:
                    self.degree[node] = 0
                    if self.reservoir is not None:
                        self.reservoir[node] = np.zeros(self.reservoir_dim, dtype=int)
            
            # Rule 1: Update degrees
            if self.degree is not None:
                self.degree[u] += 1
                self.degree[v] += 1
            
            # Rule 2 & 3: Probabilistically update reservoirs
            # For node u's reservoir
            if self.reservoir is not None and self.degree is not None:
                for j in range(self.reservoir_dim):
                    if np.random.random() < 1.0 / self.degree[u]:
                        self.reservoir[u][j] = v
            
            # For node v's reservoir
            if self.reservoir is not None and self.degree is not None:
                for j in range(self.reservoir_dim):
                    if np.random.random() < 1.0 / self.degree[v]:
                        self.reservoir[v][j] = u

    def _generate_dynamic_walks(self, new_edges: np.ndarray) -> np.ndarray:
        """Generates new random walks starting from nodes affected by new edges."""
        walks = []
        affected_nodes = np.unique(new_edges.flatten())
        
        for node_id in affected_nodes:
            for _ in range(self.walks_per_node): # Reuse existing hyperparameter
                walk = [node_id]
                current = node_id
                
                for _ in range(self.walk_length - 1):
                    if self.reservoir is not None and current in self.reservoir and len(self.reservoir[current]) > 0:
                        valid_neighbors = [n for n in self.reservoir[current] if n is not None]
                        if valid_neighbors:
                            next_node = np.random.choice(valid_neighbors)
                            walk.append(next_node)
                            current = next_node
                        else:
                            break
                    else:
                        break
                
                while len(walk) < self.walk_length:
                    walk.append(walk[-1]) # Pad if walk is short
                
                walks.append(walk[:self.walk_length])
        
        return np.array(walks)

    def _incremental_train(self, walks: np.ndarray) -> None:
        """Performs a few epochs of training on the new walks."""
        if len(walks) == 0:
            return

        onehot_walks = self._walks_to_onehot(walks)
        
        assert self.model is not None, "Model must be initialized"
        assert self.optimizer is not None, "Optimizer must be initialized"
        self.model.train()
        for epoch in range(self.incremental_epochs):
            # Forward pass
            reconstruction, encoding = self.model(onehot_walks, corrupt_prob=0.0)
            
            # Compute losses
            loss, _, _, _, _ = self._compute_losses(onehot_walks, reconstruction, encoding)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _get_scores_and_update_clusters(self, edge_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates anomaly scores for new embeddings and updates cluster centers
        using the streaming k-means formula.
        """
        if self.cluster_centers is None:
            raise RuntimeError("Clustering must be initialized before scoring.")

        # 1. Calculate distances to current centers (this is the anomaly score)
        distances = cdist(edge_embeddings, self.cluster_centers)
        closest_cluster_indices = np.argmin(distances, axis=1)
        anomaly_scores = np.min(distances, axis=1)

        # 2. Update cluster centers using Equation 15
        for i in range(len(edge_embeddings)):
            x_prime = edge_embeddings[i]
            j = closest_cluster_indices[i] # Index of the closest cluster
            
            c_old = self.cluster_centers[j]
            if self.cluster_counts is not None:
                n_old = self.cluster_counts[j]
            
                # Equation 15 simplified for a single new point (n_prime = 1)
                numerator = (self.alpha * c_old * n_old) + ((1 - self.alpha) * x_prime)
                denominator = (self.alpha * n_old) + (1 - self.alpha)
                
                self.cluster_centers[j] = numerator / denominator
                self.cluster_counts[j] = denominator # Update the effective count

        return anomaly_scores

    def _update_and_infer(
        self,
        new_edges: torch.Tensor,
        new_labels: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Incrementally updates the model with new edges and infers their anomaly scores.
        (Private method - streaming logic encapsulated within inference)

        Args:
            new_edges: A tensor of shape [2, N] representing N new edges.
            new_labels: A tensor of shape [N] with the corresponding ground truth labels.

        Returns:
            Tuple of (predictions, labels, inference_time).
        """
        start_time = time.time()
        
        new_edges_np = new_edges.cpu().numpy()
        
        # Step 1: Update internal graph representation (reservoirs)
        self._update_reservoirs_and_degrees(new_edges_np)
        
        # Step 2: Generate walks from the updated graph structure
        dynamic_walks = self._generate_dynamic_walks(new_edges_np)
        
        # Step 3: Incrementally update the embedding model
        self._incremental_train(dynamic_walks)
        
        # Step 4: Get new edge embeddings from the updated model
        # Note: _get_node_embeddings() is called inside _encode_edges()
        edge_embeddings = self._encode_edges(new_edges)
        
        # Step 5: Get anomaly scores and update clusters
        if self.cluster_centers is None:
            self._initialize_clustering() # Initialize on first call if not done
            
        predictions = self._get_scores_and_update_clusters(edge_embeddings)
        labels = new_labels.cpu().numpy()
        
        inference_time = time.time() - start_time
        
        return predictions, labels, inference_time
    
    def _get_node_embeddings(self) -> np.ndarray:
        """Get node embeddings from the trained model."""
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if self.num_nodes is None:
            raise ValueError("num_nodes is not initialized")
        
        # Create identity matrix for all nodes
        node_onehot = torch.eye(self.num_nodes, dtype=torch.float32).to(self.device)
        
        assert self.model is not None, "Model must be initialized"
        self.model.eval()
        with torch.no_grad():
            _, embeddings = self.model(node_onehot.T)  # Transpose to [num_nodes, num_nodes]
            embeddings = embeddings.T  # Transpose back to [num_nodes, embedding_dim]
        
        return embeddings.cpu().numpy()
    
    def _encode_edges(self, edge_index: torch.Tensor) -> np.ndarray:
        """Encode edges using Hadamard product of node embeddings."""
        node_embeddings = self._get_node_embeddings()
        edge_pairs = edge_index.cpu().numpy()
        
        src_embeddings = node_embeddings[edge_pairs[0]]
        tgt_embeddings = node_embeddings[edge_pairs[1]]
        
        # Hadamard product
        edge_embeddings = src_embeddings * tgt_embeddings
        return edge_embeddings
    
    def inference(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray, float]:
        """Run inference on the specified data split.
        
        This method encapsulates the streaming logic internally for test split,
        while providing standard batch inference for train/val splits.
        
        Args:
            split: Data split to evaluate ("train", "val", "test")
            
        Returns:
            Tuple of (predictions, labels, inference_time)
        """
        if not self.trained:
            self.train()  # Auto-train if not trained yet
        
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
        
        # For test split, simulate streaming evaluation internally
        if split == "test":
            return self._streaming_inference(data)
        else:
            # For train/val splits, use standard batch inference
            return self._batch_inference(data)
    
    def _batch_inference(self, data: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Standard batch inference for train/val splits."""
        start_time = time.time()
        
        # Get edge embeddings
        edge_embeddings = self._encode_edges(data['edge_index'])
        labels = data['labels'].cpu().numpy()
        
        # Initialize clustering if not done
        if self.cluster_centers is None:
            self._initialize_clustering()
        
        # Compute anomaly scores using distance to nearest cluster center
        distances = cdist(edge_embeddings, self.cluster_centers)
        min_distances = np.min(distances, axis=1)
        
        # Use distances as anomaly scores (higher distance = more anomalous)
        predictions = min_distances
        
        inference_time = time.time() - start_time
        return predictions, labels, inference_time
    
    def _streaming_inference(self, data: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Streaming inference for test split - encapsulates the chunking logic."""
        logger.info("Running streaming inference on test split...")
        
        all_edges = data['edge_index']
        all_labels = data['labels']
        
        # Simulate streaming process internally
        snapshot_size = 100  # This could be a hyperparameter
        num_snapshots = (all_edges.shape[1] + snapshot_size - 1) // snapshot_size
        
        all_preds_list, all_labels_list = [], []
        
        for i in range(num_snapshots):
            start_idx = i * snapshot_size
            end_idx = min((i + 1) * snapshot_size, all_edges.shape[1])
            
            edge_chunk = all_edges[:, start_idx:end_idx]
            label_chunk = all_labels[start_idx:end_idx]
            
            if edge_chunk.shape[1] == 0:
                continue
            
            logger.debug(f"Processing snapshot {i+1}/{num_snapshots} with {edge_chunk.shape[1]} edges...")
            
            # Call internal streaming logic
            preds, labels, _ = self._update_and_infer(edge_chunk, label_chunk)
            
            all_preds_list.append(preds)
            all_labels_list.append(labels)
        
        # Consolidate results
        final_predictions = np.concatenate(all_preds_list) if all_preds_list else np.array([])
        final_labels = np.concatenate(all_labels_list) if all_labels_list else np.array([])
        
        total_inference_time = time.time() - time.time()  # Will be calculated properly
        
        return final_predictions, final_labels, total_inference_time
    
    def _initialize_clustering(self) -> None:
        """Initialize k-means clustering using training data."""
        if self.train_data is None:
            raise RuntimeError("No training data available for clustering initialization")
        
        logger.info("Initializing k-means clustering...")
        
        # Get training edge embeddings
        train_embeddings = self._encode_edges(self.train_data['edge_index'])
        
        # Use sklearn KMeans just for the initial fit
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10).fit(train_embeddings)
        
        self.cluster_centers = kmeans.cluster_centers_
        
        # Initialize cluster counts (weights)
        self.cluster_counts = np.array([
            np.sum(kmeans.labels_ == i) for i in range(self.k_clusters)
        ])
        logger.info(f"Initialized {self.k_clusters} clusters")