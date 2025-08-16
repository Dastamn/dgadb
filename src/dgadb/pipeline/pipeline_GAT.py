#!/usr/bin/env python3
"""
GAT Pipeline for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This pipeline implements GAT-based anomaly detection using PyTorch Geometric.
It learns node embeddings through graph attention mechanisms and uses them for edge-level
anomaly detection via a downstream classifier.
"""

import logging
import numpy as np
import sys
import time

import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.GAT.GAT_main import GATModel
from src.dgadb.pipeline.load_graph import load_graph
from src.dgadb.utils.load_config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    
    # Configuration
    dataset_name = "bitcoin-alpha"  # Can be changed to any dataset
    
    logger.info(f"Starting GAT pipeline for dataset: {dataset_name}")
    
    # Load configuration
    try:
        config = load_config(dataset_name)
        logger.info(f"Loaded config: {config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Use default config if loading fails
        config = {
            "snapshot_size": 1000,
            "train_ratio": 0.70,
            "anomaly_ratio": 0.01,
            "temporal_window_size": 100
        }
        logger.info(f"Using default config: {config}")
    
    # Device configuration - Force CPU to avoid CUDA issues
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model hyperparameters
    hyperparams = {
        # GAT parameters
        "hidden_channels": 64,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.1,
        "v2": True,  # Use GATv2Conv
        
        # Training parameters
        "num_epoch": 5,  # Reduced for testing
        "learning_rate": 0.01,
        
        # Classifier parameters
        "classifier_solver": "lbfgs",
        "classifier_max_iter": 1000,
    }
    
    logger.info(f"Model hyperparameters: {hyperparams}")
    
    try:
        # Step 1: Load and preprocess graph data
        logger.info("Step 1: Loading and preprocessing graph data...")
        start_time = time.time()
        
        graph = load_graph(dataset_name)
        
        logger.info(f"Data loading and preprocessing took {time.time() - start_time:.2f} seconds")
        
        # Step 2: Initialize GAT model
        logger.info("Step 2: Initializing GAT model...")
        start_time = time.time()
        
        model = GATModel(
            device=device,
            hyperparams=hyperparams,
            epoch_evaluation_metric=roc_auc_score
        )
        
        model.setup(graph)
        logger.info(f"Model initialization took {time.time() - start_time:.2f} seconds")
        
        # Step 6: Train GAT model
        logger.info("Step 6: Training GAT model...")
        start_time = time.time()
        
        model.train()
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Step 7: Evaluate model
        logger.info("Step 7: Evaluating model...")
        
        # Test evaluation
        test_preds, test_labels, test_inf_time = model.inference("test")
        test_auc = roc_auc_score(test_labels, test_preds)
        
        logger.info("Test Results:")
        logger.info(f"  - ROC-AUC Score: {test_auc:.4f}")
        logger.info(f"  - Inference Time: {test_inf_time:.2f} seconds")
        logger.info(f"  - Test Edges: {len(test_labels)}")
        logger.info(f"  - Test Anomalies: {int(test_labels.sum())}")
        
        # Train evaluation (for comparison)
        train_preds, train_labels, train_inf_time = model.inference("train")
        train_auc = roc_auc_score(train_labels, train_preds)
        
        logger.info("Train Results (for comparison):")
        logger.info(f"  - ROC-AUC Score: {train_auc:.4f}")
        logger.info(f"  - Inference Time: {train_inf_time:.2f} seconds")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Model: GAT")
        logger.info(f"Device: {device}")
        logger.info(f"Total Edges: {graph.num_edges}")
        logger.info(f"Training Time: {training_time:.2f} seconds")
        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}")
        logger.info("=" * 60)
        
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()