#!/usr/bin/env python3
"""
Hyperparameter search space definitions for DGADB models.

This module provides predefined search spaces for different model types
using Ray Tune's search space API.
"""

from typing import Dict, Any, Optional
import logging

try:
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_search_space(model_type: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a search space for a specific model type.
    
    Args:
        model_type: Type of model ("EvolveGCN", "GCN", "GAT", "GraphSAGE", etc.)
        dataset_name: Optional dataset name for dataset-specific tuning
        
    Returns:
        Dictionary defining the search space for Ray Tune
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune is not available. Please install ray[tune]")
    
    model_type = model_type.lower()
    
    if model_type == "evolvegcn":
        return _get_evolve_gcn_search_space(dataset_name)
    elif model_type == "gcn":
        return _get_gcn_search_space(dataset_name)
    elif model_type == "gat":
        return _get_gat_search_space(dataset_name)
    elif model_type == "graphsage":
        return _get_graphsage_search_space(dataset_name)
    elif model_type == "node2vec":
        return _get_node2vec_search_space(dataset_name)
    elif model_type == "jodie":
        return _get_jodie_search_space(dataset_name)
    elif model_type == "netwalk":
        return _get_netwalk_search_space(dataset_name)
    elif model_type == "gtn":
        return _get_gtn_search_space(dataset_name)
    elif model_type == "slade":
        return _get_slade_search_space(dataset_name)
    elif model_type == "taddy":
        return _get_taddy_search_space(dataset_name)
    elif model_type == "strgnn":
        return _get_strgnn_search_space(dataset_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _get_evolve_gcn_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for EvolveGCN model."""
    return {
        "embedding_dim": tune.choice([16, 32, 64, 128, 256]),
        "hidden_dim": tune.choice([32, 64, 128, 256, 512]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "dropout": tune.uniform(0.1, 0.7),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200, 300]),
        "decoder_epochs": tune.choice([50, 100, 200]),
        "decoder_learning_rate": tune.loguniform(1e-4, 1e-1),
        "remove_self_loops": tune.choice([True, False])
    }


def _get_gcn_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for GCN model."""
    return {
        "embedding_dim": tune.choice([64, 128, 256, 512]),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "dropout": tune.uniform(0.1, 0.7),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200, 300]),
        "decoder_epochs": tune.choice([50, 100, 200]),
        "decoder_learning_rate": tune.loguniform(1e-4, 1e-1),
        "remove_self_loops": tune.choice([True, False])
    }


def _get_gat_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for GAT model."""
    return {
        "hidden_channels": tune.choice([32, 64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3]),
        "num_heads": tune.choice([1, 2, 4, 8]),
        "dropout": tune.uniform(0.1, 0.7),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "v2": tune.choice([True, False])
    }


def _get_graphsage_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for GraphSAGE model."""
    return {
        "hidden_channels": tune.choice([64, 128, 256, 512]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "dropout": tune.uniform(0.1, 0.7),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "decoder_epochs": tune.choice([50, 100, 200]),
        "decoder_learning_rate": tune.loguniform(1e-4, 1e-1)
    }


def _get_node2vec_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for Node2Vec model."""
    return {
        "embedding_dim": tune.choice([64, 128, 256, 512]),
        "walk_length": tune.choice([10, 20, 30, 40]),
        "context_size": tune.choice([5, 10, 15, 20]),
        "walks_per_node": tune.choice([5, 10, 15, 20]),
        "p": tune.loguniform(0.5, 2.0),
        "q": tune.loguniform(0.5, 2.0),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([5, 10, 20, 30]),
        "batch_size": tune.choice([64, 128, 256, 512])
    }


def _get_jodie_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for JODIE model."""
    return {
        "embedding_dim": tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([64, 128, 256]),
        "decoder_epochs": tune.choice([50, 100, 200]),
        "decoder_learning_rate": tune.loguniform(1e-4, 1e-1)
    }


def _get_netwalk_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for NetWalk model."""
    return {
        "embedding_dim": tune.choice([32, 64, 128, 256]),
        "walk_length": tune.choice([3, 5, 7, 10]),
        "walks_per_node": tune.choice([3, 5, 10, 15]),
        "reservoir_dim": tune.choice([5, 10, 15, 20]),
        "learning_rate": tune.loguniform(1e-3, 1e-1),
        "num_epoch": tune.choice([2, 5, 10, 20]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "gamma": tune.loguniform(10.0, 1000.0),
        "lamb": tune.loguniform(1e-4, 1e-2),
        "beta": tune.uniform(0.5, 2.0),
        "rho": tune.uniform(0.1, 0.9),
        "k_clusters": tune.choice([3, 5, 7, 10]),
        "alpha": tune.uniform(0.1, 0.9)
    }


def _get_gtn_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for GTN model."""
    return {
        "embedding_dim": tune.choice([64, 128, 256]),
        "num_channels": tune.choice([1, 2, 3, 4]),
        "num_layers": tune.choice([1, 2, 3]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "decoder_epochs": tune.choice([50, 100, 200]),
        "decoder_learning_rate": tune.loguniform(1e-4, 1e-1)
    }


def _get_slade_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for SLADE model."""
    return {
        "embedding_dim": tune.choice([64, 128, 256]),
        "memory_dim": tune.choice([64, 128, 256, 512]),
        "temporal_dim": tune.choice([32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "num_neighbors": tune.choice([10, 20, 30, 40])
    }


def _get_taddy_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for TADDY model."""
    return {
        "embedding_dim": tune.choice([64, 128, 256]),
        "hidden_dim": tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "window_size": tune.choice([3, 5, 7, 10])
    }


def _get_strgnn_search_space(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Search space for StrGNN model."""
    return {
        "embedding_dim": tune.choice([64, 128, 256]),
        "hidden_dim": tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3])
    }


# Utility function to get model-specific default ranges
def get_default_range(param_name: str, model_type: str) -> Any:
    """
    Get default range for a specific parameter and model type.
    
    Args:
        param_name: Name of the parameter
        model_type: Type of model
        
    Returns:
        Default range or value for the parameter
    """
    defaults = {
        "embedding_dim": tune.choice([64, 128, 256]),
        "hidden_dim": tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_epoch": tune.choice([50, 100, 200]),
        "dropout": tune.uniform(0.1, 0.5),
        "batch_size": tune.choice([32, 64, 128, 256])
    }
    
    return defaults.get(param_name, tune.choice([None]))
