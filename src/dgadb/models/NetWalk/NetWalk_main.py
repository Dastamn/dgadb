#!/usr/bin/env python3
"""
NetWalk Model for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This module implements NetWalk-based anomaly detection using TensorFlow.
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
import tensorflow as tf
from typing import Any, Callable, Optional, Tuple, Dict, List
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
from collections import Counter
import random

from src.dgadb.storage.graph import Graph

logger = logging.getLogger(__name__)

# Disable TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
        
        # TensorFlow session and model
        self.sess: Optional[tf.Session] = None
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
        
        # Move graph to device (though we'll work with numpy for TensorFlow)
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
        
        # Build TensorFlow model
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
        """Build the TensorFlow clique embedding model."""
        logger.info("Building TensorFlow clique embedding model...")
        
        # Disable eager execution for TensorFlow v1 compatibility
        tf.compat.v1.disable_eager_execution()

        # Reset default graph
        tf.compat.v1.reset_default_graph()
        
        # Input placeholder
        self.data_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[self.num_nodes, None], name='data')
        self.corrupt_prob = tf.compat.v1.placeholder(tf.float32, [1])
        
        # Build autoencoder
        dimensions = [self.num_nodes, self.embedding_dim]
        self.loss, self.clique_loss, self.ae_loss, self.kl_loss, self.weight_decay = self._build_clique_embedding_loss(dimensions)
        
        # Optimizer
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        # Initialize session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
        self.model_built = True
        logger.info("TensorFlow model built successfully")
    
    def _build_clique_embedding_loss(self, dimensions: List[int]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Build the clique embedding loss function."""
        
        # Autoencoder part
        x = tf.cast(self.data_placeholder, tf.float32)
        current_input = x * (1 - self.corrupt_prob) + self._corrupt(x) * self.corrupt_prob
        
        weight_decay_J = 0
        
        # Encoder
        encoder_weights = []
        for layer_i, n_output in enumerate(dimensions[1:]):
            n_input = int(current_input.get_shape()[0])
            
            W = tf.compat.v1.get_variable(f"encoder_W_{layer_i}",
                              shape=[n_output, n_input],
                              initializer=tf.compat.v1.uniform_unit_scaling_initializer(factor=1.0, seed=24))
            b = tf.Variable(tf.zeros([1, n_output]), name=f"encoder_b_{layer_i}")
            
            encoder_weights.append(W)
            output = tf.nn.sigmoid(tf.transpose(a=tf.transpose(a=tf.matmul(W, current_input)) + b))
            current_input = output
            weight_decay_J += (self.lamb / 2.0) * tf.reduce_mean(input_tensor=W ** 2)
        
        encoder_out = current_input
        
        # Decoder
        for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
            n_input = int(current_input.get_shape()[0])
            
            W = tf.compat.v1.get_variable(f"decoder_W_{layer_i}",
                              shape=[n_output, n_input],
                              initializer=tf.compat.v1.uniform_unit_scaling_initializer(factor=1.0, seed=24))
            b = tf.Variable(tf.zeros([1, n_output]), name=f"decoder_b_{layer_i}")
            
            output = tf.nn.sigmoid(tf.transpose(a=tf.transpose(a=tf.matmul(W, current_input)) + b))
            current_input = output
            weight_decay_J += (self.lamb / 2.0) * tf.reduce_mean(input_tensor=W ** 2)
        
        reconstruction = current_input
        
        # Autoencoder loss
        ae_loss = tf.reduce_mean(input_tensor=tf.square(reconstruction - x))
        
        # Sparsity constraint (KL divergence)
        rhohats = tf.reduce_mean(input_tensor=tf.transpose(a=encoder_out), axis=0)
        kl_loss = tf.reduce_mean(input_tensor=
            self.rho * tf.math.log(self.rho / (rhohats + 1e-8)) +
            (1 - self.rho) * tf.math.log((1 - self.rho) / (1 - rhohats + 1e-8))
        )
        
        # Clique loss
        phi = np.ones((self.walk_length, self.walk_length)) - np.eye(self.walk_length)
        L = tf.cast(tf.constant(np.diag(np.sum(phi, axis=1)) - phi), tf.float32)
        
        trans_code = tf.transpose(a=encoder_out)
        trans_code = tf.reshape(trans_code, [-1, self.walk_length, dimensions[-1]])
        t_trans_code = tf.transpose(a=trans_code, perm=[0, 2, 1])
        
        left = tf.einsum('aij,jk->aik', t_trans_code, L)
        mul = tf.einsum('aij,ajk->aik', left, trans_code)
        trace_mul = tf.linalg.trace(mul)
        clique_loss = tf.reduce_mean(input_tensor=trace_mul)
        
        # Total loss
        total_loss = clique_loss + self.gamma * ae_loss + self.beta * kl_loss + weight_decay_J
        
        # Store encoder output for inference
        self.encoder_out = encoder_out
        
        return total_loss, clique_loss, ae_loss, kl_loss, weight_decay_J
    
    def _corrupt(self, x: tf.Tensor) -> tf.Tensor:
        """Add noise for denoising autoencoder."""
        return tf.add(x, tf.random.uniform(shape=tf.shape(x), minval=0, maxval=0.1))
    
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
            for _ in range(self.walks_per_node):
                walk = [node_id]
                current = node_id
                
                for _ in range(self.walk_length - 1):
                    if current in self.reservoir and len(self.reservoir[current]) > 0:
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
    
    def _walks_to_onehot(self, walks: np.ndarray) -> np.ndarray:
        """Convert walks to one-hot encoded format."""
        walk_mat = walks.flatten()
        rows = walk_mat
        cols = np.arange(len(rows))
        data = np.ones(len(rows))
        
        # Create sparse matrix and convert to dense
        from scipy.sparse import coo_matrix
        coo = coo_matrix((data, (rows, cols)), shape=(self.num_nodes, len(rows)))
        return coo.toarray().astype(np.float32)
    
    def initial_train(self) -> None:
        """Train the NetWalk model using clique embedding on the initial graph."""
        if not self.model_built:
            raise RuntimeError("Model not built. Call setup() first.")
        
        logger.info(f"Starting NetWalk initial training for {self.num_epochs} epochs...")
        
        # Generate walks
        walks = self._generate_walks()
        onehot_walks = self._walks_to_onehot(walks)
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Batch the data
            batch_size = self.batch_size * self.walk_length
            num_batches = onehot_walks.shape[1] // batch_size
            
            total_loss = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, onehot_walks.shape[1])
                batch_data = onehot_walks[:, start_idx:end_idx]
                
                feed_dict = {
                    self.data_placeholder: batch_data,
                    self.corrupt_prob: [0.0]  # No corruption for now
                }
                
                if self.sess is not None:
                    loss_val, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                    total_loss += loss_val
            
            # Log progress and validation score
            if (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                avg_loss = total_loss / max(num_batches, 1)
                log_msg = f"Epoch {epoch + 1}: loss={avg_loss:.4f}"
                
                # Add validation score if validation data is available
                if self.val_data is not None:
                    try:
                        val_preds, val_labels, _ = self.inference("val")
                        val_auc = self.epoch_evaluation_metric(val_labels, val_preds)
                        log_msg += f", val_auc={val_auc:.4f}"
                    except Exception as e:
                        logger.warning(f"Could not compute validation score: {e}")
                
                logger.info(log_msg)
        
        self.trained = True
        logger.info("NetWalk initial training completed")

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
        
        for epoch in range(self.incremental_epochs):
            # Simplified batching for smaller incremental data
            feed_dict = {
                self.data_placeholder: onehot_walks,
                self.corrupt_prob: [0.0]
            }
            if self.sess is not None:
                self.sess.run(self.optimizer, feed_dict=feed_dict)

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

    def update_and_infer(
        self,
        new_edges: torch.Tensor,
        new_labels: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Incrementally updates the model with new edges and infers their anomaly scores.

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
        
        # Create identity matrix for all nodes
        if self.num_nodes is None:
            raise ValueError("num_nodes is not initialized")
        node_onehot = np.eye(self.num_nodes, dtype=np.float32)
        
        feed_dict = {
            self.data_placeholder: node_onehot,
            self.corrupt_prob: [0.0]
        }
        
        if self.sess is None:
            raise ValueError("TensorFlow session is not initialized")
        embeddings = self.sess.run(tf.transpose(a=self.encoder_out), feed_dict=feed_dict)
        return embeddings
    
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
        
        Args:
            split: Data split to evaluate ("train", "val", "test")
            
        Returns:
            Tuple of (predictions, labels, inference_time)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
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
    
    def _initialize_clustering(self) -> None:
        """Initialize k-means clustering using training data."""
        if self.train_data is None:
            raise RuntimeError("No training data available for clustering initialization")
        
        logger.info("Initializing k-means clustering...")
        
        # Get training edge embeddings
        train_embeddings = self._encode_edges(self.train_data['edge_index'])
        
        # Use sklearn KMeans just for the initial fit
        if self.train_data is not None:
            train_embeddings = self._encode_edges(self.train_data['edge_index'])
            kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10).fit(train_embeddings)
            
            self.cluster_centers = kmeans.cluster_centers_
            
            # Initialize cluster counts (weights)
            self.cluster_counts = np.array([
                np.sum(kmeans.labels_ == i) for i in range(self.k_clusters)
            ])
            logger.info(f"Initialized {self.k_clusters} clusters")
    
    def __del__(self):
        """Clean up TensorFlow session."""
        if hasattr(self, 'sess') and self.sess is not None:
            self.sess.close()