import logging
import time
from collections.abc import Callable
from typing import Any, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

from src.dgadb.storage.graph import Graph
from src.dgadb.models.common import EdgeDecoder, train_edge_decoder, inference_with_decoder

logger = logging.getLogger(__name__)


class Node2VecModel:
    """Node2Vec-based anomaly detection model for DGADB.
    
    This class implements a Node2Vec-based approach for dynamic graph anomaly detection.
    It learns node embeddings using random walks and then uses these embeddings for
    edge-level anomaly detection through a downstream classifier.
    
    Attributes:
        device (torch.device): Device to run computations on.
        epoch_evaluation_metric (callable): Function to evaluate model performance.
        node2vec (Node2Vec): PyTorch Geometric Node2Vec model.
        classifier (LogisticRegression): Downstream classifier for anomaly detection.
        edge_index (torch.Tensor): Graph edge indices.
        node_embeddings (torch.Tensor): Learned node embeddings.
        
    Args:
        device (torch.device): Device to run computations on.
        hyperparams (dict): Dictionary containing model hyperparameters including
            embedding_dim, walk_length, context_size, walks_per_node, p, q, etc.
        epoch_evaluation_metric (callable): Function that takes (labels, predictions)
            and returns a scalar evaluation metric.
    """
    
    def __init__(
        self,
        device: torch.device,
        hyperparams: dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        """Initialize Node2VecModel with device, hyperparameters, and evaluation metric.
        
        Args:
            device (torch.device): Device to run computations on.
            hyperparams (dict): Dictionary containing model hyperparameters.
            epoch_evaluation_metric (callable): Function for evaluation during training.
        """
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        
        logger.info(f"Initializing Node2VecModel with device={self.device} and hyperparams={hyperparams}")
        
        # Node2Vec hyperparameters
        self.embedding_dim = hyperparams.get("embedding_dim", 128)
        self.walk_length = hyperparams.get("walk_length", 20)
        self.context_size = hyperparams.get("context_size", 10)
        self.walks_per_node = hyperparams.get("walks_per_node", 10)
        self.p = hyperparams.get("p", 1.0)
        self.q = hyperparams.get("q", 1.0)
        self.num_negative_samples = hyperparams.get("num_negative_samples", 1)
        self.sparse = hyperparams.get("sparse", True)
        
        # Training hyperparameters
        self.num_epoch = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.01)
        self.batch_size = hyperparams.get("batch_size", 128)
        
        # Classifier hyperparameters (deprecated - kept for backward compatibility)
        self.classifier_solver = hyperparams.get("classifier_solver", "lbfgs")
        self.classifier_max_iter = hyperparams.get("classifier_max_iter", 1000)
        
        # Decoder parameters (new configurable parameters)
        self.decoder_epochs = hyperparams.get("decoder_epochs", 100)
        self.decoder_learning_rate = hyperparams.get("decoder_learning_rate", 0.01)
        
        # Initialize components
        self.node2vec: Optional[Node2Vec] = None
        self.decoder: Optional[EdgeDecoder] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        # Data containers
        self.edge_index: Optional[torch.Tensor] = None
        self.node_embeddings: Optional[torch.Tensor] = None
        self.train_data: Optional[Dict[str, torch.Tensor]] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        self.has_val: bool = False
        self.num_nodes: Optional[int] = None
        
    def setup(self, graph: Graph) -> None:
        """Set up data processing and initialize the Node2Vec model.
        
        Args:
            graph (Graph): Graph object containing node and edge data.
        """
        logger.info("Setting up Node2VecModel...")
        
        graph.to(self.device)
        
        self.num_nodes = graph.num_nodes
        
        # Check for validation split
        self.has_val = hasattr(graph, "e_val_mask")
        
        logger.info(f"Graph has {self.num_nodes} nodes and {graph.num_edges} edges")
        
        # Create edge index for the entire graph (used for Node2Vec training)
        self.edge_index = graph.e_pairs
        
        # Use edge index as-is from the graph (preprocessing handled by pipeline)
        logger.info(f"Using edge index with {self.edge_index.shape[1]} edges")
        
        # Prepare data splits for evaluation
        self._prepare_data_splits(graph)
        
        # Initialize Node2Vec model
        self.node2vec = Node2Vec(
            edge_index=self.edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=self.num_negative_samples,
            num_nodes=self.num_nodes,
            sparse=self.sparse
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.SparseAdam(
            list(self.node2vec.parameters()), 
            lr=self.learning_rate
        )
        
        logger.info("Node2VecModel setup completed")
    
    def _prepare_data_splits(self, graph: Graph) -> None:
        """Prepare train/test/val data splits for evaluation."""
        
        # Prepare training data
        train_mask = graph.e_train_mask
        if train_mask.any():
            self.train_data = {
                "edge_index": graph.e_pairs[:, train_mask],
                "labels": graph.e_label[train_mask].long(),
            }
        
        # Prepare test data
        test_mask = graph.e_test_mask
        if test_mask.any():
            self.test_data = {
                "edge_index": graph.e_pairs[:, test_mask],
                "labels": graph.e_label[test_mask].long(),
            }
        
        # Prepare validation data (if available)
        if self.has_val:
            val_mask = graph.e_val_mask
            if val_mask.any():
                self.val_data = {
                    "edge_index": graph.e_pairs[:, val_mask],
                    "labels": graph.e_label[val_mask].long(),
                }
        
        train_count = len(self.train_data['labels']) if self.train_data else 0
        test_count = len(self.test_data['labels']) if self.test_data else 0
        val_count = len(self.val_data['labels']) if self.val_data and self.has_val else 0
        
        logger.info(f"Prepared data splits - Train: {train_count}, Test: {test_count}, Val: {val_count}")
    
    def _ensure_setup(self) -> None:
        """Ensure setup() has been called before using the model."""
        if any(attr is None for attr in [self.node2vec, self.optimizer, self.edge_index, 
                                        self.train_data, self.test_data]):
            raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")
    
    def train(self) -> None:
        """Train the Node2Vec model.
        
        Performs Node2Vec embedding learning through random walk sampling and
        skip-gram optimization. After embedding learning, trains a downstream
        classifier for anomaly detection.
        """
        self._ensure_setup()
        logger.info(f"Starting Node2Vec training for {self.num_epoch} epochs...")
        
        # Train Node2Vec embeddings
        self.node2vec.train()
        
        for epoch in range(self.num_epoch):
            total_loss = 0
            num_batches = 0
            
            # Create data loader for Node2Vec training
            loader = self.node2vec.loader(batch_size=self.batch_size, shuffle=True)
            
            for pos_rw, neg_rw in tqdm(loader, desc=f"Epoch {epoch+1}/{self.num_epoch}"):
                self.optimizer.zero_grad()
                loss = self.node2vec.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            # Evaluate on validation set every 10 epochs
            if (epoch + 1) % 10 == 0:
                split = "val" if self.has_val else "test"
                pred_score, labels, _ = self.inference(split=split)
                epoch_score = self.epoch_evaluation_metric(labels, pred_score)
                
                logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, {split}_auc={epoch_score:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        logger.info("Node2Vec training completed")
    
    def _get_node_embeddings(self) -> torch.Tensor:
        """Get node embeddings from the trained Node2Vec model."""
        if self.node_embeddings is None:
            self.node2vec.eval()
            with torch.no_grad():
                # Get embeddings for all nodes
                all_nodes = torch.arange(self.num_nodes, device=self.device)
                self.node_embeddings = self.node2vec(all_nodes)
        
        return self.node_embeddings
    
    def _get_edge_embeddings(self, edge_index: torch.Tensor) -> np.ndarray:
        """Get edge embeddings by combining node embeddings.
        
        Args:
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            
        Returns:
            np.ndarray: Edge embeddings [num_edges, embedding_dim * 2]
        """
        node_embeddings = self._get_node_embeddings()
        
        src_embeddings = node_embeddings[edge_index[0]]
        tgt_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=1)
        
        return edge_embeddings.cpu().numpy()
    
    def _train_decoder(self) -> None:
        """Train the downstream decoder for anomaly detection."""
        if self.train_data is None:
            raise RuntimeError("No training data available")
        
        logger.info("Training downstream decoder...")
        
        # Get node embeddings
        if self.node_embeddings is None:
            self._get_node_embeddings()
        
        # Initialize decoder
        self.decoder = EdgeDecoder(embedding_dim=self.embedding_dim).to(self.device)
        
        # Train decoder using the common training function with configurable parameters
        train_edge_decoder(
            decoder=self.decoder,
            node_embeddings=self.node_embeddings,
            train_edge_index=self.train_data["edge_index"],
            train_labels=self.train_data["labels"].float(),
            num_epochs=self.decoder_epochs,
            learning_rate=self.decoder_learning_rate,
            device=self.device
        )
        
        logger.info("Decoder training completed")
    
    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray, float]:
        """Run inference on the specified data split.
        
        Args:
            split (str): Data split to evaluate on ("train", "val", "test").
            
        Returns:
            tuple: A tuple containing:
                - pred_score (np.ndarray): Predicted anomaly scores
                - labels (np.ndarray): Ground truth binary labels  
                - inf_time (float): Inference time in seconds
                
        Raises:
            ValueError: If the specified split is not available.
        """
        self._ensure_setup()
        start_time = time.time()
        
        # Select data split
        if split == "train":
            data = self.train_data
        elif split == "val":
            if not self.has_val:
                raise ValueError("No validation split available.")
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if data is None:
            raise ValueError(f"No data available for split: {split}")
        
        # Train decoder if not already trained
        if self.decoder is None:
            self._train_decoder()
        
        # Get node embeddings if not available
        if self.node_embeddings is None:
            self._get_node_embeddings()
        
        # Use the common decoder for inference
        pred_proba, labels, inf_time = inference_with_decoder(
            self.decoder, self.node_embeddings, data["edge_index"], data["labels"].float()
        )
        
        return pred_proba, labels, inf_time