#!/usr/bin/env python3
"""
YAML-based hyperparameter configuration loader.

This module loads hyperparameter search spaces from YAML configuration files
and converts them to Ray Tune search spaces.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging

try:
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_search_space_from_yaml(model_type: str, config_dir: str = "configs/tuning") -> Dict[str, Any]:
    """
    Load hyperparameter search space from YAML configuration file.
    
    Args:
        model_type: Type of model (e.g., "evolve_gcn", "gcn", "gat")
        config_dir: Directory containing YAML configuration files
        
    Returns:
        Dictionary with Ray Tune search space configuration
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune is not available. Please install ray[tune]")
    
    # Construct config file path
    config_file = os.path.join(config_dir, f"{model_type.lower()}.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    # Load YAML configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert YAML configuration to Ray Tune search space
    search_space = {}
    for param_name, param_config in config.items():
        search_space[param_name] = _convert_config_to_tune(param_config)
    
    return search_space


def _convert_config_to_tune(param_config: Dict[str, Any]) -> Any:
    """
    Convert YAML parameter configuration to Ray Tune search space object.
    
    Args:
        param_config: Dictionary with parameter configuration
        
    Returns:
        Ray Tune search space object
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune is not available")
    
    param_type = param_config.get("type")
    
    if param_type == "choice":
        # Convert string "true"/"false" to actual booleans
        values = param_config["values"]
        processed_values = []
        for val in values:
            if isinstance(val, str):
                if val.lower() == "true":
                    processed_values.append(True)
                elif val.lower() == "false":
                    processed_values.append(False)
                else:
                    processed_values.append(val)
            else:
                processed_values.append(val)
        return tune.choice(processed_values)
    elif param_type == "uniform":
        return tune.uniform(float(param_config["min"]), float(param_config["max"]))
    elif param_type == "loguniform":
        return tune.loguniform(float(param_config["min"]), float(param_config["max"]))
    elif param_type == "quniform":
        return tune.quniform(float(param_config["min"]), float(param_config["max"]), float(param_config.get("q", 1)))
    elif param_type == "qloguniform":
        return tune.qloguniform(float(param_config["min"]), float(param_config["max"]), float(param_config.get("q", 1)))
    elif param_type == "randint":
        return tune.randint(int(param_config["min"]), int(param_config["max"]))
    elif param_type == "qrandint":
        return tune.qrandint(int(param_config["min"]), int(param_config["max"]), int(param_config.get("q", 1)))
    elif param_type == "randn":
        return tune.randn(float(param_config["mean"]), float(param_config["std"]))
    elif param_type == "qrandn":
        return tune.qrandn(float(param_config["mean"]), float(param_config["std"]), float(param_config.get("q", 1)))
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def get_search_space(model_type: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get search space for a model type, with optional dataset-specific adjustments.
    
    Args:
        model_type: Type of model
        dataset_name: Optional dataset name for dataset-specific tuning
        
    Returns:
        Dictionary with Ray Tune search space
    """
    try:
        # First try to load from YAML config
        search_space = load_search_space_from_yaml(model_type)
    except FileNotFoundError:
        # Fall back to programmatic search space if YAML not found
        logger.warning(f"No YAML config found for {model_type}, using programmatic search space")
        from src.dgadb.tuning.core.search_spaces import create_search_space
        search_space = create_search_space(model_type, dataset_name)
    
    # Apply dataset-specific adjustments if needed
    if dataset_name:
        search_space = _apply_dataset_specific_adjustments(search_space, dataset_name)
    
    return search_space


def _apply_dataset_specific_adjustments(search_space: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Apply dataset-specific adjustments to the search space.
    
    Args:
        search_space: Original search space
        dataset_name: Dataset name
        
    Returns:
        Adjusted search space
    """
    # Dataset-specific adjustments can be implemented here
    # For example, smaller models for larger datasets
    
    dataset_adjustments = {
        "bitcoin-alpha": {
            # Smaller embedding dimensions for bitcoin datasets
            "embedding_dim": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "num_epoch": tune.choice([50, 100])
        },
        "bitcoin-otc": {
            "embedding_dim": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "num_epoch": tune.choice([50, 100])
        },
        "reddit": {
            # Larger models for Reddit dataset
            "embedding_dim": tune.choice([128, 256, 512]),
            "hidden_dim": tune.choice([256, 512, 1024])
        },
        "wiki": {
            "embedding_dim": tune.choice([128, 256, 512]),
            "hidden_dim": tune.choice([256, 512, 1024])
        }
    }
    
    if dataset_name in dataset_adjustments:
        adjustments = dataset_adjustments[dataset_name]
        for param_name, adjusted_space in adjustments.items():
            if param_name in search_space:
                search_space[param_name] = adjusted_space
                logger.info(f"Applied dataset-specific adjustment for {param_name} on {dataset_name}")
    
    return search_space


def list_available_configs(config_dir: str = "configs/tuning") -> list:
    """
    List available hyperparameter configuration files.
    
    Args:
        config_dir: Directory containing YAML configuration files
        
    Returns:
        List of available model configurations
    """
    if not os.path.exists(config_dir):
        return []
    
    configs = []
    for file in os.listdir(config_dir):
        if file.endswith('.yaml'):
            configs.append(file[:-5])  # Remove .yaml extension
    
    return sorted(configs)


def create_config_template(model_type: str, config_dir: str = "configs/tuning") -> str:
    """
    Create a template YAML configuration file for a model type.
    
    Args:
        model_type: Type of model
        config_dir: Directory to save the template
        
    Returns:
        Path to the created template file
    """
    os.makedirs(config_dir, exist_ok=True)
    
    template = {
        "embedding_dim": {
            "type": "choice",
            "values": [64, 128, 256]
        },
        "hidden_dim": {
            "type": "choice",
            "values": [64, 128, 256]
        },
        "learning_rate": {
            "type": "loguniform",
            "min": 1e-4,
            "max": 1e-1
        },
        "num_epoch": {
            "type": "choice",
            "values": [50, 100, 200]
        },
        "dropout": {
            "type": "uniform",
            "min": 0.1,
            "max": 0.5
        }
    }
    
    config_file = os.path.join(config_dir, f"{model_type.lower()}.yaml")
    
    with open(config_file, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    
    return config_file
