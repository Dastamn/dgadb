#!/usr/bin/env python3
"""
Core hyperparameter tuning functionality using Ray Tune.

This module provides the base HyperparameterTuner class that integrates
with Ray Tune for distributed hyperparameter optimization.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, Callable, List, Tuple
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search import ConcurrencyLimiter
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from src.dgadb.storage.graph import Graph
from src.dgadb.storage.temporal_graph_data import TemporalGraphData
from src.dgadb.utils.load_config import load_config

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Base class for hyperparameter tuning using Ray Tune.
    
    This class provides a framework for tuning hyperparameters of DGADB models
    using distributed optimization with Ray Tune and integration with Optuna/Hyperopt.
    """
    
    def __init__(
        self,
        model_class: Callable,
        dataset_name: str,
        metric: str = "auc",
        num_samples: int = 10,
        max_concurrent_trials: int = 4,
        gpus_per_trial: float = 0.5,
        cpus_per_trial: float = 2.0,
        results_dir: str = "./tuning_results"
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_class: The model class to tune (e.g., EvolveGCNModel)
            dataset_name: Name of the dataset to use for tuning
            metric: Evaluation metric to optimize ("auc", "f1", "precision", "recall")
            num_samples: Number of hyperparameter configurations to try
            max_concurrent_trials: Maximum number of concurrent trials
            gpus_per_trial: GPU resources per trial (0 for CPU-only)
            cpus_per_trial: CPU resources per trial
            results_dir: Directory to save tuning results
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune is not available. Please install ray[tune]")
            
        self.model_class = model_class
        self.dataset_name = dataset_name
        self.metric = metric
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        logger.info(f"Initialized HyperparameterTuner for {model_class.__name__} on {dataset_name}")
    
    def _trainable_function(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trainable function for Ray Tune.
        
        This function is executed for each hyperparameter configuration.
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() and self.gpus_per_trial > 0 else "cpu")
            
            # Load and preprocess data
            from src.dgadb.pipeline.pipeline import Pipeline
            pipeline = Pipeline.from_config(f"{self.dataset_name}-example")
            processed_data = pipeline.run()
            
            temporal_graph = processed_data.to_temporal_graph()
            from src.dgadb.data.builder import build_graph_from_temporal
            graph = build_graph_from_temporal(temporal_graph)
            
            # Initialize model with current hyperparameters
            model = self.model_class(
                device=device,
                hyperparams=config,
                epoch_evaluation_metric=roc_auc_score
            )
            
            # Setup model
            model.setup(temporal_graph, graph)
            
            # Train model
            model.train()
            
            # Evaluate on validation set
            val_preds, val_labels, _ = model.inference("val")
            
            if val_preds.size == 0:
                auc_score = 0.0
            else:
                auc_score = roc_auc_score(val_labels, val_preds)
            
            # Return metrics
            return {
                self.metric: auc_score,
                "loss": 1 - auc_score,  # For minimization
                "done": True
            }
            
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return {
                self.metric: 0.0,
                "loss": 1.0,
                "done": True,
                "error": str(e)
            }
    
    def tune(
        self,
        search_space: Dict[str, Any],
        scheduler: Optional[Any] = None,
        search_algorithm: Optional[Any] = None,
        time_budget_s: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            search_space: Dictionary defining the hyperparameter search space
            scheduler: Ray Tune scheduler for early stopping
            search_algorithm: Search algorithm (Optuna, Hyperopt, etc.)
            time_budget_s: Time budget in seconds for the tuning process
            
        Returns:
            Dictionary containing best hyperparameters and results
        """
        if scheduler is None:
            scheduler = ASHAScheduler(
                max_t=100,
                grace_period=10,
                reduction_factor=2
            )
        
        # Configure reporter
        reporter = CLIReporter(
            metric_columns=[self.metric, "loss", "training_iteration"]
        )
        
        # Configure resources
        resources_per_trial = {
            "cpu": self.cpus_per_trial,
            "gpu": self.gpus_per_trial
        }
        
        # Run tuning
        analysis = tune.run(
            self._trainable_function,
            config=search_space,
            metric="loss",
            mode="min",
            num_samples=self.num_samples,
            resources_per_trial=resources_per_trial,
            scheduler=scheduler,
            search_alg=search_algorithm,
            progress_reporter=reporter,
            storage_path=self.results_dir,
            time_budget_s=time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            verbose=1
        )
        
        # Get best results
        best_trial = analysis.best_trial
        best_config = analysis.best_config
        best_metric = analysis.best_result[self.metric]
        
        # Save results
        results = {
            "best_config": best_config,
            "best_metric": best_metric,
            "best_trial_id": best_trial.trial_id,
            "all_trials": analysis.results,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_class.__name__,
            "dataset": self.dataset_name
        }
        
        # Save to file
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_class.__name__}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Tuning completed. Best {self.metric}: {best_metric:.4f}")
        logger.info(f"Best configuration: {best_config}")
        logger.info(f"Results saved to: {results_file}")
        
        return results
    
    def get_default_search_space(self) -> Dict[str, Any]:
        """
        Get default search space for the model.
        
        Returns:
            Dictionary with default search space configuration
        """
        # This should be implemented by model-specific subclasses
        raise NotImplementedError("Subclasses should implement get_default_search_space")
    
    def create_study(self, study_name: Optional[str] = None) -> Any:
        """
        Create an Optuna study for advanced hyperparameter optimization.
        
        Args:
            study_name: Name for the study
            
        Returns:
            Optuna study object
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
            
            if study_name is None:
                study_name = f"{self.model_class.__name__}_{self.dataset_name}"
                
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize" if self.metric in ["auc", "f1", "precision", "recall"] else "minimize",
                sampler=TPESampler()
            )
            
            return study
            
        except ImportError:
            logger.warning("Optuna not available. Please install optuna for advanced optimization.")
            return None


# Utility function for easy tuning
def tune_hyperparameters(
    model_class: Callable,
    dataset_name: str,
    search_space: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter tuning.
    
    Args:
        model_class: The model class to tune
        dataset_name: Name of the dataset
        search_space: Optional custom search space
        **kwargs: Additional arguments for HyperparameterTuner
        
    Returns:
        Tuning results
    """
    tuner = HyperparameterTuner(model_class, dataset_name, **kwargs)
    
    if search_space is None:
        search_space = tuner.get_default_search_space()
    
    return tuner.tune(search_space)
