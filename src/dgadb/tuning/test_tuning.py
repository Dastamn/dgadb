#!/usr/bin/env python3
"""
Test script to run actual hyperparameter tuning on GCN model.
"""

import os
import sys
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.dgadb.tuning.core.tune_core import HyperparameterTuner
from src.dgadb.models.GCN.GCN_main import GCNModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_gcn_tuning():
    """Test hyperparameter tuning on GCN model."""
    print("=== Testing GCN Hyperparameter Tuning ===")
    
    try:
        # Create tuner instance
        tuner = HyperparameterTuner(
            model_class=GCNModel,
            dataset_name="bitcoin-alpha",
            num_samples=3,  # Small number for testing
            max_concurrent_trials=1,
            gpus_per_trial=0,  # CPU-only for testing
            results_dir=os.path.abspath("./test_tuning_results")
        )
        
        # Get search space
        from src.dgadb.tuning.configs.model_configs import get_search_space
        search_space = get_search_space("gcn")
        
        print(f"Search space parameters: {list(search_space.keys())}")
        
        # Run a quick test with just 2 trials
        print("Starting hyperparameter tuning with 2 trials...")
        
        # Modify search space to be very small for testing
        test_search_space = {
            "embedding_dim": search_space["embedding_dim"],
            "learning_rate": search_space["learning_rate"],
            "num_epoch": search_space["num_epoch"]
        }
        
        # Run tuning
        results = tuner.tune(
            search_space=test_search_space
        )
        
        print("Tuning completed!")
        print(f"Best AUC: {results['best_metric']:.4f}")
        print(f"Best config: {results['best_config']}")
        
        # Clean up
        import shutil
        if os.path.exists("./test_tuning_results"):
            shutil.rmtree("./test_tuning_results")
            
    except Exception as e:
        print(f"Error during tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gcn_tuning()
