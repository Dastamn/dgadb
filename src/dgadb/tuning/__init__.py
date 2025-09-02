"""
Hyperparameter tuning module for DGADB.

This module provides hyperparameter optimization capabilities using Ray Tune
with integration for Optuna and Hyperopt optimization algorithms.
Supports YAML-based configuration for hyperparameter search spaces.
"""

from .core.tune_core import HyperparameterTuner
from .configs.model_configs import (
    get_search_space,
    load_search_space_from_yaml,
    list_available_configs,
    create_config_template
)
from .cli.tune import main as tune_main

__all__ = [
    'HyperparameterTuner',
    'get_search_space',
    'load_search_space_from_yaml',
    'list_available_configs',
    'create_config_template',
    'tune_main'
]

__version__ = "0.1.0"
