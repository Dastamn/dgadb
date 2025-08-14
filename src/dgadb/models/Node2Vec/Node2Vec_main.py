import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

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
        
        # Classifier hyperparameters
        self.classifier_solver = hyperparams.get("classifier_solver", "lbfgs")
        self.classifier_max_iter = hyperparams.get("classifier_max_iter", 1000)
        
        # Initialize components
        self.node2vec: Node2Vec | None = None
        self.classifier: LogisticRegression | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        
        # Data containers
        self.edge_index: torch.Tensor | None = None
        self.node_embeddings: torch.Tensor | None = None
        self.train_data: dict | None = None
        self.test_data: dict | None = None
        self.val_data: dict | None = None
        self.has_val: bool = False
        self.num_nodes: int | None = None
        
    def setup(self, df: pl.DataFrame) -> None:
        """Set up data processing and initialize the Node2Vec model.
        
        Processes the input DataFrame into Node2Vec format, creates edge indices,
        and initializes the Node2Vec model and optimizer.
        
        Args:
            df (pl.DataFrame): Input DataFrame containing columns: src, tgt, 
                label, train_mask, test_mask, and optionally val_mask.
        """
        logger.info("Setting up Node2VecModel...")
        
        # Validate required columns
        required_cols = ["src", "tgt", "label", "train_mask", "test_mask"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check for validation split
        self.has_val = "val_mask" in df.columns
        
        # Get all unique nodes and create node mapping
        all_nodes = np.unique(np.concatenate([df["src"].to_numpy(), df["tgt"].to_numpy()]))
        self.num_nodes = len(all_nodes)
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
        
        logger.info(f"Graph has {self.num_nodes} nodes and {len(df)} edges")
        
        # Create edge index for the entire graph (used for Node2Vec training)
        src_mapped = df["src"].map_elements(lambda x: node_mapping[x], return_dtype=pl.Int64)
        tgt_mapped = df["tgt"].map_elements(lambda x: node_mapping[x], return_dtype=pl.Int64)
        
        # Create edge index tensor with CUDA error handling
        try:
            self.edge_index = torch.stack([
                torch.tensor(src_mapped.to_numpy(), dtype=torch.long),
                torch.tensor(tgt_mapped.to_numpy(), dtype=torch.long)
            ]).to(self.device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning(f"CUDA error encountered: {e}")
                logger.info("Falling back to CPU device")
                self.device = torch.device("cpu")
                self.edge_index = torch.stack([
                    torch.tensor(src_mapped.to_numpy(), dtype=torch.long),
                    torch.tensor(tgt_mapped.to_numpy(), dtype=torch.long)
                ]).to(self.device)
            else:
                raise e
        
        # Make graph undirected by adding reverse edges
        reverse_edge_index = torch.stack([self.edge_index[1], self.edge_index[0]])
        self.edge_index = torch.cat([self.edge_index, reverse_edge_index], dim=1)
        
        # Remove duplicate edges
        self.edge_index = torch.unique(self.edge_index, dim=1)
        
        logger.info(f"Created undirected edge index with {self.edge_index.shape[1]} edges")
        
        # Prepare data splits for evaluation
        self._prepare_data_splits(df, node_mapping)
        
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
    
    def _prepare_data_splits(self, df: pl.DataFrame, node_mapping: dict) -> None:
        """Prepare train/test/val data splits for evaluation."""
        
        def prepare_split(mask_col: str) -> dict:
            split_df = df.filter(pl.col(mask_col))
            if len(split_df) == 0:
                return None
                
            src_mapped = split_df["src"].map_elements(lambda x: node_mapping[x], return_dtype=pl.Int64)
            tgt_mapped = split_df["tgt"].map_elements(lambda x: node_mapping[x], return_dtype=pl.Int64)
            
            return {
                "edge_index": torch.stack([
                    torch.tensor(src_mapped.to_numpy(), dtype=torch.long),
                    torch.tensor(tgt_mapped.to_numpy(), dtype=torch.long)
                ]).to(self.device),
                "labels": torch.tensor(split_df["label"].to_numpy(), dtype=torch.long).to(self.device)
            }
        
        self.train_data = prepare_split("train_mask")
        self.test_data = prepare_split("test_mask")
        
        if self.has_val:
            self.val_data = prepare_split("val_mask")
        
        logger.info(f"Prepared data splits - Train: {len(self.train_data['labels']) if self.train_data else 0}, "
                   f"Test: {len(self.test_data['labels']) if self.test_data else 0}, "
                   f"Val: {len(self.val_data['labels']) if self.val_data and self.has_val else 0}")
    
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
    
    def _train_classifier(self) -> None:
        """Train the downstream classifier for anomaly detection."""
        if self.train_data is None:
            raise RuntimeError("No training data available")
        
        # Get edge embeddings for training data
        train_embeddings = self._get_edge_embeddings(self.train_data["edge_index"])
        train_labels = self.train_data["labels"].cpu().numpy()
        
        # Train logistic regression classifier
        self.classifier = LogisticRegression(
            solver=self.classifier_solver,
            max_iter=self.classifier_max_iter,
            random_state=42
        )
        
        logger.info("Training downstream classifier...")
        self.classifier.fit(train_embeddings, train_labels)
        
        # Log training performance
        train_pred_proba = self.classifier.predict_proba(train_embeddings)[:, 1]
        train_auc = roc_auc_score(train_labels, train_pred_proba)
        logger.info(f"Training AUC: {train_auc:.4f}")
    
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
        
        # Train classifier if not already trained
        if self.classifier is None:
            self._train_classifier()
        
        # Get embeddings and predictions
        embeddings = self._get_edge_embeddings(data["edge_index"])
        labels = data["labels"].cpu().numpy()
        
        # Get prediction probabilities
        pred_proba = self.classifier.predict_proba(embeddings)[:, 1]
        
        inf_time = time.time() - start_time
        
        return pred_proba, labels, inf_time