import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import negative_sampling
import numpy as np
from typing import Any, Callable, Optional
import logging
import time

from src.dgadb.storage.graph import Graph
from src.dgadb.models.common import EdgeDecoder, train_edge_decoder, inference_with_decoder
from src.dgadb.preprocessing.structural import make_undirected_tensor

logger = logging.getLogger(__name__)

class GraphSAGEModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        hyperparams: dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        super().__init__()
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric

        # --- NEW HYPERPARAMS for GraphSAGE ---
        self.in_channels = -1
        self.hidden_channels = hyperparams.get("hidden_channels", 128)
        self.out_channels = hyperparams.get("out_channels", 128)
        self.num_layers = hyperparams.get("num_layers", 2)
        self.dropout = hyperparams.get("dropout", 0.5)

        # Training hyperparameters
        self.num_epoch = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.01)

        # Classifier hyperparameters
        self.classifier_solver = hyperparams.get("classifier_solver", "lbfgs")
        self.classifier_max_iter = hyperparams.get("classifier_max_iter", 1000)

        # Initialize components
        self.graphsage: Optional[GraphSAGE] = None
        self.decoder: Optional[EdgeDecoder] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Data containers
        self.x: Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.node_embeddings: Optional[torch.Tensor] = None
        self.num_nodes: Optional[int] = None
        self.train_data: Optional[dict[str, Any]] = None
        self.val_data: Optional[dict[str, Any]] = None
        self.test_data: Optional[dict[str, Any]] = None

    def _ensure_setup(self):
        if self.graphsage is None or self.optimizer is None or self.x is None or self.edge_index is None:
            raise RuntimeError("The model has not been set up. Please call setup() first.")

    def setup(self, graph: Graph) -> None:
        logger.info("Setting up GraphSAGEModel...")
        graph.to(self.device)
        self.num_nodes = graph.num_nodes

        if graph.n_feat is not None:
            self.x = graph.n_feat.float()
            self.in_channels = self.x.shape[1]
            logger.info(f"Using provided node features with dimension {self.in_channels}")
        else:
            logger.warning("No node features found. Creating learnable embeddings as a fallback.")
            self.in_channels = self.hidden_channels
            self.node_emb = nn.Embedding(self.num_nodes, self.in_channels).to(self.device)
            self.x = self.node_emb.weight

        self.edge_index = graph.e_pairs
        # Apply structural preprocessing to make graph undirected and remove duplicates
        self.edge_index = make_undirected_tensor(self.edge_index)
        logger.info(f"Applied structural preprocessing: undirected graph with {self.edge_index.shape[1]} edges")

        self._prepare_data_splits(graph)

        self.graphsage = GraphSAGE(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        params = list(self.graphsage.parameters())
        if hasattr(self, 'node_emb'):
            params += list(self.node_emb.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        logger.info("GraphSAGEModel setup completed")

    def train(self) -> None:
        self._ensure_setup()
        logger.info(f"Starting GraphSAGE training for {self.num_epoch} epochs...")
        
        train_pos_edge_index = self.train_data["edge_index"].to(self.device)

        for epoch in range(self.num_epoch):
            self.graphsage.train()
            self.optimizer.zero_grad()
            
            z = self.graphsage(self.x, self.edge_index)
            
            pos_src_emb = z[train_pos_edge_index[0]]
            pos_tgt_emb = z[train_pos_edge_index[1]]
            pos_logits = (pos_src_emb * pos_tgt_emb).sum(dim=1)

            neg_edge_index = negative_sampling(
                edge_index=self.edge_index,
                num_nodes=self.num_nodes,
                num_neg_samples=train_pos_edge_index.shape[1],
            )
            neg_src_emb = z[neg_edge_index[0]]
            neg_tgt_emb = z[neg_edge_index[1]]
            neg_logits = (neg_src_emb * neg_tgt_emb).sum(dim=1)

            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits)
            ])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: loss={loss.item():.4f}")
        
        logger.info("GraphSAGE training completed")
        self.node_embeddings = self.graphsage(self.x, self.edge_index).detach()

    def _get_node_embeddings(self) -> torch.Tensor:
        if self.node_embeddings is None:
            self.graphsage.eval()
            with torch.no_grad():
                self.node_embeddings = self.graphsage(self.x, self.edge_index)
        return self.node_embeddings

    def _get_edge_embeddings(self, edge_index: torch.Tensor) -> torch.Tensor:
        node_embeddings = self._get_node_embeddings()
        source_node_embeddings = node_embeddings[edge_index[0]]
        destination_node_embeddings = node_embeddings[edge_index[1]]
        return torch.cat([source_node_embeddings, destination_node_embeddings], dim=1)

    def train_decoder(self) -> None:
        """Trains the downstream decoder on the training data."""
        self._ensure_setup()
        logger.info("Training the downstream decoder...")
        
        # Get node embeddings
        node_embeddings = self._get_node_embeddings()
        
        # Initialize decoder
        self.decoder = EdgeDecoder(embedding_dim=self.out_channels).to(self.device)
        
        # Train decoder using the common training function
        train_edge_decoder(
            decoder=self.decoder,
            node_embeddings=node_embeddings,
            train_edge_index=self.train_data["edge_index"],
            train_labels=self.train_data["edge_label"],
            num_epochs=100,
            learning_rate=0.01,
            device=self.device
        )
        
        logger.info("Downstream decoder training completed")

    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run inference on the specified data split.

        Args:
            split (str): Data split to evaluate on ("train", "val", "test").
            
        Returns:
            tuple: A tuple containing (anomaly_scores, true_labels, inference_time).
        """
        self._ensure_setup()
        start_time = time.time()

        if self.decoder is None:
            # This check is now the pipeline's responsibility.
            raise RuntimeError("Decoder has not been trained. Call train_decoder() first.")
        
        # 1. Select the correct data split based on the 'split' argument
        if split == "train":
            data = self.train_data
        elif split == "val":
            if self.val_data is None:
                raise ValueError("Validation split not available for this dataset.")
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: '{split}'")
        
        logger.info(f"Performing inference on '{split}' split...")
        
        # 2. Get node embeddings and use common decoder for inference
        node_embeddings = self._get_node_embeddings()
        
        # 3. Use the common decoder for inference
        anomaly_scores, true_labels, inference_time = inference_with_decoder(
            self.decoder, node_embeddings, data["edge_index"], data["edge_label"]
        )
        
        logger.info(f"Inference on '{split}' split completed in {inference_time:.2f}s")
        
        # 4. Return the standard tuple
        return anomaly_scores, true_labels, inference_time

    def _prepare_data_splits(self, graph: Graph):
        # Use the masks to create the data splits
        self.train_data = {
            "edge_index": graph.e_pairs[:, graph.e_train_mask],
            "edge_label": graph.e_label[graph.e_train_mask],
        }
        if hasattr(graph, "e_val_mask"):
            self.val_data = {
                "edge_index": graph.e_pairs[:, graph.e_val_mask],
                "edge_label": graph.e_label[graph.e_val_mask],
            }
        else:
            self.val_data = None
        self.test_data = {
            "edge_index": graph.e_pairs[:, graph.e_test_mask],
            "edge_label": graph.e_label[graph.e_test_mask],
        }