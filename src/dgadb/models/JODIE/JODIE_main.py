#!/usr/bin/env python3
"""
JODIE Model for Dynamic Graph Anomaly Detection Benchmark (DGADB)

Implementation of the JODIE (Joint Dynamic User-Item Embeddings) model
for temporal interaction networks and anomaly detection.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks.
S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on 
Knowledge Discovery and Data Mining (KDD), 2019.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Tuple, Dict

from src.dgadb.storage.graph import Graph
from src.dgadb.models.common import EdgeDecoder, train_edge_decoder, inference_with_decoder

logger = logging.getLogger(__name__)


class JODIE_Core(nn.Module):
    """
    Core implementation of the JODIE model architecture.
    It includes the coupled RNNs for user and item embedding updates,
    the temporal projection layer, and the next-item embedding prediction layer.
    """
    
    def __init__(self, num_features: int, embedding_dim: int, num_users: int, num_items: int):
        super().__init__()
        
        # 1. Parameter setup
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items

        # 2. Coupled Recurrent Neural Networks (Update Operation)
        # As per the paper, RNNs update dynamic embeddings.
        # Inputs: [previous_embedding, time_difference, interaction_features]
        rnn_input_size = embedding_dim + 1 + num_features
        self.user_rnn = nn.RNNCell(rnn_input_size, embedding_dim)
        self.item_rnn = nn.RNNCell(rnn_input_size, embedding_dim)

        # 3. Temporal Projection Layer
        # Implements: u_D(t + ∆) = (1 + w) * u(t), where w is learned from ∆.
        # This corresponds to `context_convert` in the reference code.
        self.projection_layer = nn.Linear(1, embedding_dim, bias=False)

        # 4. Next Item Embedding Prediction Layer
        # Predicts the next item's concatenated (dynamic + static) embedding.
        # Inputs: [projected_user_dynamic, user_static, prev_item_dynamic, prev_item_static]
        prediction_input_dim = (embedding_dim * 2) + num_users + num_items
        prediction_output_dim = embedding_dim + num_items  # Predicts (dynamic + static) item embedding
        self.prediction_layer = nn.Linear(prediction_input_dim, prediction_output_dim)

    def forward(self, 
                user_dynamic_embedding: torch.Tensor, 
                item_dynamic_embedding: torch.Tensor,
                user_timediff: torch.Tensor,
                item_timediff: torch.Tensor,
                features: torch.Tensor,
                select: str) -> torch.Tensor:
        """
        A unified forward pass to handle both user and item updates.
        This follows the pattern of the reference code's forward method.
        """
        if select == 'user_update':
            # Input to user RNN: [item_embedding, user_timediff, features]
            rnn_input = torch.cat([item_dynamic_embedding, user_timediff, features], dim=1)
            return self.user_rnn(rnn_input, user_dynamic_embedding)
        
        elif select == 'item_update':
            # Input to item RNN: [user_embedding, item_timediff, features]
            rnn_input = torch.cat([user_dynamic_embedding, item_timediff, features], dim=1)
            return self.item_rnn(rnn_input, item_dynamic_embedding)
        
        else:
            raise ValueError("Invalid 'select' mode for JODIE_Core forward pass.")

    def project_user_embedding(self, user_dynamic_embedding: torch.Tensor, user_timediff: torch.Tensor) -> torch.Tensor:
        """
        Projects a user's embedding forward in time.
        u_D(t + ∆) = (1 + time_projection_vector) * u(t)
        """
        time_projection_vector = self.projection_layer(user_timediff)
        return user_dynamic_embedding * (1 + time_projection_vector)
        
    def predict_item_embedding(self, 
                               projected_user_dynamic: torch.Tensor, 
                               user_static: torch.Tensor,
                               prev_item_dynamic: torch.Tensor,
                               prev_item_static: torch.Tensor) -> torch.Tensor:
        """
        Predicts the next item's full embedding (dynamic + static).
        """
        prediction_input = torch.cat([projected_user_dynamic, user_static, prev_item_dynamic, prev_item_static], dim=1)
        return self.prediction_layer(prediction_input)


class JODIEModel:
    """JODIE-based anomaly detection model for DGADB."""

    def __init__(
        self,
        device: torch.device,
        hyperparams: Dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        """Initialize JODIEModel with device, hyperparameters, and evaluation metric."""
        # 1. Standard initialization from the DGADB template
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        
        # 2. Extract hyperparameters
        self.embedding_dim = hyperparams.get("embedding_dim", 128)
        self.num_epochs = hyperparams.get("num_epoch", 50)
        self.learning_rate = hyperparams.get("learning_rate", 1e-3)
        self.regularization_lambda = hyperparams.get("regularization_lambda", 0.001)  # For temporal smoothness
        
        self.decoder_epochs = hyperparams.get("decoder_epochs", 100)
        self.decoder_learning_rate = hyperparams.get("decoder_learning_rate", 0.01)
        
        # 3. Initialize placeholders
        self.model: Optional[JODIE_Core] = None
        self.decoder: Optional[EdgeDecoder] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        # Dynamic embeddings for all users and items
        self.user_dynamic_embeddings: Optional[nn.Embedding] = None
        self.item_dynamic_embeddings: Optional[nn.Embedding] = None

        # Static embeddings (will be identity matrices)
        self.user_static_embeddings: Optional[torch.Tensor] = None
        self.item_static_embeddings: Optional[torch.Tensor] = None
        
        # Pre-processed training data
        self.sorted_train_data: Optional[Dict[str, torch.Tensor]] = None
        # Other data splits (val, test)
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        
        logger.info(f"Initializing JODIEModel with device={device} and hyperparams={hyperparams}")

    def setup(self, graph: Graph) -> None:
        """Set up data, initialize embeddings, and prepare the model."""
        logger.info("Setting up JODIEModel...")
        
        graph.to(self.device)
        self.num_nodes = graph.num_nodes  # Assuming users and items are in a contiguous node ID space

        # Assumption: The DGADB framework can distinguish between users and items
        # and provide their counts. If not, this needs to be inferred.
        # For simplicity, we assume all nodes can be users or items.
        self.num_users = self.num_nodes
        self.num_items = self.num_nodes

        logger.info(f"Graph has {self.num_nodes} nodes and {graph.num_edges} edges")

        # 4. Prepare data splits (standard) and sort training data by time
        # This must be done first to determine the actual feature dimensions
        self._prepare_data_splits(graph)

        # 1. Initialize core JODIE model (after we know the feature dimensions)
        self.model = JODIE_Core(
            num_features=self.num_features,
            embedding_dim=self.embedding_dim,
            num_users=self.num_users,
            num_items=self.num_items,
        ).to(self.device)

        # 2. Initialize dynamic embeddings as learnable nn.Embedding layers
        self.user_dynamic_embeddings = nn.Embedding(self.num_users, self.embedding_dim, sparse=False).to(self.device)
        self.item_dynamic_embeddings = nn.Embedding(self.num_items, self.embedding_dim, sparse=False).to(self.device)
        # Initialize with random normalized vectors as in reference code
        nn.init.normal_(self.user_dynamic_embeddings.weight)
        nn.init.normal_(self.item_dynamic_embeddings.weight)

        # 3. Initialize static embeddings (identity matrix, not trained)
        self.user_static_embeddings = torch.eye(self.num_users, device=self.device)
        self.item_static_embeddings = torch.eye(self.num_items, device=self.device)
        
        # 5. Initialize optimizer for all learnable parameters
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + 
            list(self.user_dynamic_embeddings.parameters()) + 
            list(self.item_dynamic_embeddings.parameters()),
            lr=self.learning_rate
        )
        
        logger.info("JODIEModel setup completed")

    def _prepare_data_splits(self, graph: Graph) -> None:
        """Prepare train/val/test splits and preprocess temporal information for training."""
        # Standard preparation for test and validation data (for the decoder)
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

        # JODIE-specific preparation for training data
        train_mask = graph.e_train_mask
        if not train_mask.any():
            logger.warning("No training data available")
            return

        # Check if timestamps are available
        if not hasattr(graph, 'e_timestamp') or graph.e_timestamp is None:
            logger.error("JODIE requires timestamps but graph.e_timestamp is not available")
            raise ValueError("JODIE requires temporal information (timestamps) in the graph")

        # Assumption: Graph object contains timestamps and interaction features.
        # These MUST be provided by the DGADB preprocessing pipeline.
        timestamps = graph.e_timestamp[train_mask]
        
        # Use edge features if available, otherwise use node features
        if hasattr(graph, 'e_feat') and graph.e_feat is not None:
            features = graph.e_feat[train_mask]
            self.num_features = features.shape[1]
        else:
            # Use single node features (not concatenated) to avoid dimension explosion
            edge_pairs = graph.e_pairs[:, train_mask]
            features = graph.n_feat[edge_pairs[0]]  # Use source node features only
            self.num_features = features.shape[1]
        
        # Sort all training data by timestamp to process interactions sequentially
        sorted_indices = torch.argsort(timestamps)
        
        edge_pairs = graph.e_pairs[:, train_mask][:, sorted_indices]
        users, items = edge_pairs[0], edge_pairs[1]
        
        # Calculate time differences (∆t) between consecutive interactions for each user/item
        last_user_timestamp = torch.zeros(self.num_users, device=self.device)
        last_item_timestamp = torch.zeros(self.num_items, device=self.device)
        user_timediffs = torch.zeros_like(timestamps)
        item_timediffs = torch.zeros_like(timestamps)
        
        # This loop is crucial for creating the temporal context JODIE needs
        for i in range(len(users)):
            u, v, t = users[i], items[i], timestamps[sorted_indices][i]
            user_timediffs[i] = t - last_user_timestamp[u]
            item_timediffs[i] = t - last_item_timestamp[v]
            last_user_timestamp[u] = t
            last_item_timestamp[v] = t
            
        # Store the prepared and sorted training data
        self.sorted_train_data = {
            'users': users,
            'items': items,
            'timestamps': timestamps[sorted_indices],
            'features': features[sorted_indices],
            'user_timediffs': user_timediffs.unsqueeze(1).float(),
            'item_timediffs': item_timediffs.unsqueeze(1).float(),
            'labels': graph.e_label[train_mask][sorted_indices].float()
        }
        
        # Log data split sizes
        train_count = len(self.sorted_train_data['labels']) if self.sorted_train_data else 0
        test_count = len(self.test_data['labels']) if self.test_data else 0
        val_count = len(self.val_data['labels']) if self.val_data else 0
        
        logger.info(f"Prepared data splits - Train: {train_count}, Test: {test_count}, Val: {val_count}")

    def train(self) -> None:
        """Train the JODIE model for one epoch."""
        self._ensure_setup()
        self.model.train()
        
        # Loss function from the paper (L2 distance for embeddings)
        mse_loss = nn.MSELoss()
        
        # Get all data for the epoch
        data = self.sorted_train_data
        if not data:
            logger.warning("No training data available. Skipping training.")
            return
            
        # We process interactions one-by-one in temporal order to update embeddings
        # This simulates the stream-based learning of the original JODIE model.
        total_loss = 0
        num_interactions = len(data['users'])
        
        # To get the "previous" item, we need to track it per user
        last_user_interaction_item = torch.full((self.num_users,), -1, dtype=torch.long, device=self.device)
        
        # Process only a subset for faster training (can be removed for full training)
        max_interactions = min(1000, num_interactions)  # Limit to 1000 interactions for testing
        logger.info(f"Processing {max_interactions} out of {num_interactions} interactions for this epoch")
        
        # Loop over training interactions *in temporal order*
        for i in range(max_interactions):
            if i % 100 == 0:  # Progress logging
                logger.info(f"Processing interaction {i}/{max_interactions}")
                
            self.optimizer.zero_grad()
            
            user, item = data['users'][i], data['items'][i]
            feature = data['features'][i].unsqueeze(0)
            user_timediff = data['user_timediffs'][i].unsqueeze(0)
            item_timediff = data['item_timediffs'][i].unsqueeze(0)
            
            prev_item_id = last_user_interaction_item[user]
            
            # --- Prediction Step ---
            # We can only make a prediction if the user has a previous interaction in this batch
            if prev_item_id != -1:
                # 1. Get user and previous item's current dynamic embeddings
                user_dyn_emb = self.user_dynamic_embeddings(user.unsqueeze(0))
                prev_item_dyn_emb = self.item_dynamic_embeddings(prev_item_id.unsqueeze(0))
                
                # 2. Project user embedding to the time of the current interaction
                projected_user_emb = self.model.project_user_embedding(user_dyn_emb, user_timediff)

                # 3. Predict the embedding of the *current* item
                predicted_item_full_emb = self.model.predict_item_embedding(
                    projected_user_emb,
                    self.user_static_embeddings[user].unsqueeze(0),
                    prev_item_dyn_emb,
                    self.item_static_embeddings[prev_item_id].unsqueeze(0)
                )
                
                # 4. Get the ground truth embedding for the current item
                actual_item_dyn_emb = self.item_dynamic_embeddings(item.unsqueeze(0))
                actual_item_static_emb = self.item_static_embeddings[item].unsqueeze(0)
                actual_item_full_emb = torch.cat([actual_item_dyn_emb, actual_item_static_emb], dim=1)
                
                # 5. Calculate prediction loss (Loss term 1 in the paper)
                loss = mse_loss(predicted_item_full_emb, actual_item_full_emb.detach())
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # --- Update Step ---
            # 6. Get embeddings *before* the update for regularization
            user_dyn_emb_before_update = self.user_dynamic_embeddings(user.unsqueeze(0))
            item_dyn_emb_before_update = self.item_dynamic_embeddings(item.unsqueeze(0))
            
            # 7. Update user and item embeddings using the coupled RNNs
            new_user_dyn_emb = self.model.forward(
                user_dyn_emb_before_update, item_dyn_emb_before_update,
                user_timediff, item_timediff, feature, select='user_update'
            )
            new_item_dyn_emb = self.model.forward(
                user_dyn_emb_before_update, item_dyn_emb_before_update,
                user_timediff, item_timediff, feature, select='item_update'
            )
            
            # 8. Add regularization loss (temporal smoothness, Loss terms 2 & 3)
            user_reg_loss = self.regularization_lambda * mse_loss(new_user_dyn_emb, user_dyn_emb_before_update.detach())
            item_reg_loss = self.regularization_lambda * mse_loss(new_item_dyn_emb, item_dyn_emb_before_update.detach())
            loss = loss + user_reg_loss + item_reg_loss
            
            # 9. Backpropagate and update
            if loss.item() != 0:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # 10. Manually update the embedding weights with the new values
            # This is critical for the next iteration to see the updated state
            with torch.no_grad():
                self.user_dynamic_embeddings.weight.data[user] = new_user_dyn_emb.squeeze(0).detach()
                self.item_dynamic_embeddings.weight.data[item] = new_item_dyn_emb.squeeze(0).detach()
            
            # 11. Update the last interaction tracker
            last_user_interaction_item[user] = item

        avg_loss = total_loss / max_interactions if max_interactions > 0 else 0
        logger.info(f"Epoch training loss: {avg_loss:.4f}")

    def inference(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray, float]:
        """Run inference using the common decoder."""
        self._ensure_setup()
        
        # Select data split
        if split == "test":
            data = self.test_data
        elif split == "val":
            data = self.val_data
        elif split == "train":
            # For train split, we need to create it from sorted_train_data
            if self.sorted_train_data is not None:
                data = {
                    'edge_index': torch.stack([self.sorted_train_data['users'], self.sorted_train_data['items']]),
                    'labels': self.sorted_train_data['labels']
                }
            else:
                data = None
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if data is None:
            raise ValueError(f"No data available for split: {split}")
        
        # 1. Train the downstream decoder if it hasn't been trained yet.
        if self.decoder is None:
            self._train_decoder()
            
        # 2. Get the FINAL node embeddings after training. 
        # For JODIE, these are the final states of the dynamic embeddings.
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.user_dynamic_embeddings.weight.data
        
        # 3. Use the common inference function.
        predictions, labels, inference_time = inference_with_decoder(
            self.decoder, node_embeddings, data['edge_index'], data['labels']
        )
        
        return predictions, labels, inference_time
        
    def _train_decoder(self) -> None:
        """Train the downstream EdgeDecoder for anomaly detection."""
        logger.info("Training downstream decoder...")
        
        if self.sorted_train_data is None:
            raise RuntimeError("No training data available for decoder training")
        
        self.decoder = EdgeDecoder(embedding_dim=self.embedding_dim).to(self.device)
        
        with torch.no_grad():
            node_embeddings = self.user_dynamic_embeddings.weight.data
        
        # Create train data for decoder from sorted_train_data
        train_edge_index = torch.stack([self.sorted_train_data['users'], self.sorted_train_data['items']])
        train_labels = self.sorted_train_data['labels']
        
        train_edge_decoder(
            decoder=self.decoder,
            node_embeddings=node_embeddings,
            train_edge_index=train_edge_index,
            train_labels=train_labels,
            num_epochs=self.decoder_epochs,
            learning_rate=self.decoder_learning_rate,
            device=self.device
        )
        
        logger.info("Decoder training completed")

    def _ensure_setup(self) -> None:
        """Ensure that setup() has been called before training/inference."""
        if any(attr is None for attr in [self.model, self.optimizer]):
            raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")