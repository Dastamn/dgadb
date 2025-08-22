#!/usr/bin/env python3
"""
Common components for DGADB models.

This module contains shared components that can be used across different models
to ensure consistency and avoid code duplication.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import time
import logging

logger = logging.getLogger(__name__)


class EdgeDecoder(nn.Module):
    """
    Simple linear decoder for edge prediction.
    
    Takes two node embeddings and returns a prediction of the edge between them.
    This replaces the LogisticRegression approach with a simple neural network layer.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the edge decoder.
        
        Args:
            embedding_dim (int): Dimension of node embeddings
        """
        super().__init__()
        # Linear layer that takes concatenated embeddings and outputs a single score
        self.decoder = nn.Linear(embedding_dim * 2, 1)
        
    def forward(self, src_embeddings: torch.Tensor, tgt_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            src_embeddings (torch.Tensor): Source node embeddings [num_edges, embedding_dim]
            tgt_embeddings (torch.Tensor): Target node embeddings [num_edges, embedding_dim]
            
        Returns:
            torch.Tensor: Edge scores [num_edges, 1]
        """
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=1)
        
        # Pass through linear layer
        scores = self.decoder(edge_embeddings)
        
        return scores.squeeze(-1)  # Remove last dimension to get [num_edges]


def get_edge_embeddings(node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get source and target node embeddings for given edges.
    
    Args:
        node_embeddings (torch.Tensor): Node embeddings [num_nodes, embedding_dim]
        edge_index (torch.Tensor): Edge indices [2, num_edges]
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Source and target embeddings
    """
    src_embeddings = node_embeddings[edge_index[0]]
    tgt_embeddings = node_embeddings[edge_index[1]]
    
    return src_embeddings, tgt_embeddings


def train_edge_decoder(
    decoder: EdgeDecoder,
    node_embeddings: torch.Tensor,
    train_edge_index: torch.Tensor,
    train_labels: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    device: torch.device = torch.device('cpu')
) -> None:
    """
    Train the edge decoder.
    
    Args:
        decoder (EdgeDecoder): The decoder to train
        node_embeddings (torch.Tensor): Node embeddings
        train_edge_index (torch.Tensor): Training edge indices
        train_labels (torch.Tensor): Training edge labels
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (torch.device): Device to train on
    """
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    src_embeddings, tgt_embeddings = get_edge_embeddings(node_embeddings, train_edge_index)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        scores = decoder(src_embeddings, tgt_embeddings)
        
        # Compute loss
        loss = criterion(scores, train_labels.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            logger.debug(f"Decoder training epoch {epoch + 1}: loss={loss.item():.4f}")


def inference_with_decoder(
    decoder: EdgeDecoder,
    node_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference using the edge decoder.
    
    Args:
        decoder (EdgeDecoder): Trained decoder
        node_embeddings (torch.Tensor): Node embeddings
        edge_index (torch.Tensor): Edge indices
        labels (torch.Tensor): True labels
        
    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Predictions, labels, inference time
    """
    start_time = time.time()
    
    decoder.eval()
    with torch.no_grad():
        src_embeddings, tgt_embeddings = get_edge_embeddings(node_embeddings, edge_index)
        scores = decoder(src_embeddings, tgt_embeddings)
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(scores).cpu().numpy()
        labels_np = labels.cpu().numpy()
    
    inference_time = time.time() - start_time
    
    return predictions, labels_np, inference_time