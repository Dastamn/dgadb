#!/usr/bin/env python3
"""
Command-line interface for hyperparameter tuning.

This module provides a CLI for running hyperparameter tuning experiments
using the DGADB hyperparameter tuning framework.
"""

import argparse
import logging
import sys
from typing import Optional, List

from src.dgadb.tuning.core.tune_core import HyperparameterTuner, tune_hyperparameters
from src.dgadb.tuning.configs.model_configs import (
    get_search_space,
    list_available_configs,
    create_config_template
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_class(model_name: str):
    """
    Get model class from model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class
    """
    # Import model classes dynamically
    model_mapping = {
        "evolve_gcn": "src.dgadb.models.EvolveGCN.EvolveGCN_main:EvolveGCNModel",
        "gcn": "src.dgadb.models.GCN.GCN_main:GCNModel",
        "gat": "src.dgadb.models.GAT.GAT_main:GATModel",
        "graphsage": "src.dgadb.models.GraphSAGE.GraphSAGE_main:GraphSAGEModel",
        "node2vec": "src.dgadb.models.Node2Vec.Node2Vec_main:Node2VecModel",
        "jodie": "src.dgadb.models.JODIE.JODIE_main:JODIEModel",
        "netwalk": "src.dgadb.models.NetWalk.NetWalk_main:NetWalkModel",
        "gtn": "src.dgadb.models.GTN.GTN_main:GTNModel",
        "slade": "src.dgadb.models.SLADE.SLADE_main:SLADEModel",
        "taddy": "src.dgadb.models.TADDY.TADDY_main:TADDYModel",
        "strgnn": "src.dgadb.models.StrGNN.StrGNN_main:StrGNNModel"
    }
    
    if model_name.lower() not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_mapping.keys())}")
    
    module_path, class_name = model_mapping[model_name.lower()].split(":")
    
    # Dynamic import
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    return model_class


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (optional)
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="DGADB Hyperparameter Tuning CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument("--model", required=True, help="Model to tune")
    tune_parser.add_argument("--dataset", required=True, help="Dataset to use")
    tune_parser.add_argument("--metric", default="auc", choices=["auc", "f1", "precision", "recall"],
                           help="Metric to optimize")
    tune_parser.add_argument("--num-samples", type=int, default=10,
                           help="Number of hyperparameter configurations to try")
    tune_parser.add_argument("--max-concurrent", type=int, default=4,
                           help="Maximum number of concurrent trials")
    tune_parser.add_argument("--gpus-per-trial", type=float, default=0.5,
                           help="GPU resources per trial (0 for CPU-only)")
    tune_parser.add_argument("--cpus-per-trial", type=float, default=2.0,
                           help="CPU resources per trial")
    tune_parser.add_argument("--results-dir", default="./tuning_results",
                           help="Directory to save results")
    tune_parser.add_argument("--time-budget", type=int,
                           help="Time budget in seconds for tuning")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available configurations")
    list_parser.add_argument("--config-dir", default="configs/tuning",
                           help="Directory containing configuration files")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create configuration template")
    create_parser.add_argument("--model", required=True, help="Model to create template for")
    create_parser.add_argument("--config-dir", default="configs/tuning",
                            help="Directory to save configuration template")
    
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    
    if not args:
        parser.print_help()
        return 0
    
    parsed_args = parser.parse_args(args)
    
    try:
        if parsed_args.command == "tune":
            return run_tuning(parsed_args)
        elif parsed_args.command == "list":
            return list_configs(parsed_args)
        elif parsed_args.command == "create":
            return create_template(parsed_args)
        else:
            parser.print_help()
            return 0
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def run_tuning(args) -> int:
    """Run hyperparameter tuning."""
    try:
        # Get model class
        model_class = get_model_class(args.model)
        
        # Run tuning
        results = tune_hyperparameters(
            model_class=model_class,
            dataset_name=args.dataset,
            metric=args.metric,
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
            gpus_per_trial=args.gpus_per_trial,
            cpus_per_trial=args.cpus_per_trial,
            results_dir=args.results_dir
        )
        
        logger.info(f"Tuning completed successfully!")
        logger.info(f"Best {args.metric}: {results['best_metric']:.4f}")
        logger.info(f"Best configuration saved to: {args.results_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        return 1


def list_configs(args) -> int:
    """List available configuration files."""
    configs = list_available_configs(args.config_dir)
    
    if not configs:
        print(f"No configuration files found in {args.config_dir}")
        return 0
    
    print("Available hyperparameter configurations:")
    for config in configs:
        print(f"  - {config}")
    
    return 0


def create_template(args) -> int:
    """Create configuration template."""
    try:
        config_file = create_config_template(args.model, args.config_dir)
        print(f"Created configuration template: {config_file}")
        print("Please edit this file to customize the hyperparameter search space.")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
