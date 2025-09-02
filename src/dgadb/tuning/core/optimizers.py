#!/usr/bin/env python3
"""
Optimizer integration for hyperparameter tuning.

This module provides integration with Optuna and Hyperopt optimization algorithms
for advanced hyperparameter search.
"""

import logging
from typing import Optional, Any, Dict

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import HyperbandPruner, MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import hp, tpe, rand, atpe
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.hyperopt import HyperOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_optimizer(backend: str = "optuna", algorithm: str = "tpe", **kwargs) -> Optional[Any]:
    """
    Get an optimization algorithm for hyperparameter tuning.
    
    Args:
        backend: Optimization backend ("optuna", "hyperopt", or "ray_default")
        algorithm: Algorithm to use ("tpe", "random", "cmaes", "atpe")
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Optimizer instance or None if not available
    """
    if backend.lower() == "optuna" and OPTUNA_AVAILABLE and RAY_AVAILABLE:
        return _get_optuna_optimizer(algorithm, **kwargs)
    elif backend.lower() == "hyperopt" and HYPEROPT_AVAILABLE and RAY_AVAILABLE:
        return _get_hyperopt_optimizer(algorithm, **kwargs)
    elif backend.lower() == "ray_default":
        return None  # Use Ray's default search
    else:
        logger.warning(f"Optimizer backend '{backend}' not available or not installed")
        return None


def _get_optuna_optimizer(algorithm: str = "tpe", **kwargs) -> Optional[Any]:
    """
    Get an Optuna-based optimizer for Ray Tune.
    
    Args:
        algorithm: Algorithm to use ("tpe", "random", "cmaes")
        **kwargs: Additional arguments for the sampler
        
    Returns:
        OptunaSearch instance or None
    """
    if not (OPTUNA_AVAILABLE and RAY_AVAILABLE):
        return None
    
    # Configure sampler
    if algorithm.lower() == "tpe":
        sampler = TPESampler(**kwargs)
    elif algorithm.lower() == "random":
        sampler = RandomSampler(**kwargs)
    elif algorithm.lower() == "cmaes":
        sampler = CmaEsSampler(**kwargs)
    else:
        logger.warning(f"Unknown Optuna algorithm: {algorithm}. Using TPE.")
        sampler = TPESampler(**kwargs)
    
    # Create Optuna search
    search_alg = OptunaSearch(
        sampler=sampler,
        metric="loss",
        mode="min"
    )
    
    # Limit concurrency
    return ConcurrencyLimiter(search_alg, max_concurrent=kwargs.get("max_concurrent", 4))


def _get_hyperopt_optimizer(algorithm: str = "tpe", **kwargs) -> Optional[Any]:
    """
    Get a Hyperopt-based optimizer for Ray Tune.
    
    Args:
        algorithm: Algorithm to use ("tpe", "random", "atpe")
        **kwargs: Additional arguments for the algorithm
        
    Returns:
        HyperOptSearch instance or None
    """
    if not (HYPEROPT_AVAILABLE and RAY_AVAILABLE):
        return None
    
    # Configure algorithm
    if algorithm.lower() == "tpe":
        algo = tpe.suggest
    elif algorithm.lower() == "random":
        algo = rand.suggest
    elif algorithm.lower() == "atpe":
        algo = atpe.suggest
    else:
        logger.warning(f"Unknown Hyperopt algorithm: {algorithm}. Using TPE.")
        algo = tpe.suggest
    
    # Create Hyperopt search
    search_alg = HyperOptSearch(
        algo=algo,
        metric="loss",
        mode="min",
        **kwargs
    )
    
    # Limit concurrency
    return ConcurrencyLimiter(search_alg, max_concurrent=kwargs.get("max_concurrent", 4))


def create_optuna_study(study_name: str, direction: str = "minimize", **kwargs) -> Optional[Any]:
    """
    Create an Optuna study for advanced hyperparameter optimization.
    
    Args:
        study_name: Name for the study
        direction: Optimization direction ("minimize" or "maximize")
        **kwargs: Additional arguments for the study
        
    Returns:
        Optuna study object or None
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available. Please install optuna.")
        return None
    
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=TPESampler(),
            **kwargs
        )
        return study
    except Exception as e:
        logger.error(f"Failed to create Optuna study: {e}")
        return None


def optimize_with_optuna(objective_function, n_trials: int = 100, **kwargs) -> Dict[str, Any]:
    """
    Run optimization using Optuna directly.
    
    Args:
        objective_function: Function that takes a trial and returns a value
        n_trials: Number of trials to run
        **kwargs: Additional arguments for create_study
        
    Returns:
        Dictionary with optimization results
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available for direct optimization")
        return {}
    
    study = create_optuna_study("direct_optimization", **kwargs)
    if study is None:
        return {}
    
    study.optimize(objective_function, n_trials=n_trials)
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": study.trials,
        "study": study
    }


# Utility functions for parameter suggestion
def suggest_parameter(trial, param_name: str, param_config: Dict[str, Any]) -> Any:
    """
    Suggest a parameter value using Optuna trial.
    
    Args:
        trial: Optuna trial object
        param_name: Name of the parameter
        param_config: Parameter configuration
        
    Returns:
        Suggested parameter value
    """
    if not OPTUNA_AVAILABLE:
        return param_config.get("default")
    
    param_type = param_config.get("type")
    
    if param_type == "choice":
        return trial.suggest_categorical(param_name, param_config["values"])
    elif param_type == "uniform":
        return trial.suggest_float(param_name, param_config["min"], param_config["max"])
    elif param_type == "loguniform":
        return trial.suggest_float(param_name, param_config["min"], param_config["max"], log=True)
    elif param_type == "int":
        return trial.suggest_int(param_name, param_config["min"], param_config["max"])
    else:
        return param_config.get("default")
