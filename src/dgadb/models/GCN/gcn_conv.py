#!/usr/bin/env python3
"""
GCN Convolution Layer for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This module implements a Graph Convolutional Network (GCN) convolution layer
adapted for the DGADB framework. It removes external dependencies and provides
a self-contained implementation with explicit parameters.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, Union


class GCNConv(nn.Module):
    """
    Graph Convolutional Network (GCN) convolution layer.
    
    This implementation is adapted to be self-contained and removes dependencies
    on external initialization functions and global arguments objects.
    
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool): If set to True, the layer computes A^2 instead of A.
        cached (bool): If set to True, the layer will cache the computation of the normalized adjacency matrix.
        bias (bool): If set to False, the layer will not learn an additive bias.
        remove_self_loops (bool): If set to True, will not add self-loops to the adjacency matrix.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        bias: bool = True,
        remove_self_loops: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.remove_self_loops = remove_self_loops
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # Initialize bias if requested
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Cache for normalized adjacency matrix
        self.cached_result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cached_num_edges: Optional[int] = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    
    @staticmethod
    def norm(
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
        improved: bool = False,
        dtype: Optional[torch.dtype] = None,
        remove_self_loops: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize the adjacency matrix for GCN.
        
        Args:
            edge_index: Edge indices
            num_nodes: Number of nodes
            edge_weight: Edge weights (optional)
            improved: Whether to use improved GCN normalization
            dtype: Data type for computations
            remove_self_loops: Whether to remove self-loops
            
        Returns:
            Tuple of normalized edge_index and edge weights
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        
        # Add self-loops to the adjacency matrix
        fill_value = 2.0 if improved else 1.0
        
        if not remove_self_loops:
            edge_index, tmp_edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value, num_nodes
            )
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight
        
        # Compute node degrees
        row, col = edge_index[0], edge_index[1]
        deg = degree(col, num_nodes, dtype=edge_weight.dtype if edge_weight is not None else torch.float32)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        
        # Normalize edge weights: D^(-1/2) * A * D^(-1/2)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        return edge_index, norm
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the GCN layer.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights (optional)
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform node features
        x = torch.matmul(x, self.weight)
        
        # Cache normalization if requested
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
        
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(
                edge_index, x.size(0), edge_weight, self.improved,
                x.dtype, self.remove_self_loops
            )
            if self.cached:
                self.cached_result = edge_index, norm
        else:
            edge_index, norm = self.cached_result
        
        # Message passing: aggregate neighbor features
        row, col = edge_index[0], edge_index[1]
        
        # Aggregate messages from neighbors
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.view(-1, 1))
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')