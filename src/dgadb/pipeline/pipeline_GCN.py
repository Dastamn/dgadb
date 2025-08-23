#!/usr/bin/env python3
"""
GCN Pipeline for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This pipeline implements GCN-based anomaly detection using a custom GCN implementation.
It learns node embeddings through graph convolution and uses them for edge-level
anomaly detection via a downstream classifier.
"""

import logging
import numpy as np
import sys
import time

import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.GCN.GCN_main import GCNModel
from src.dgadb.preprocessing.pipeline.pipeline import Pipeline
from src.dgadb.data.builder import build_graph_from_temporal

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
    
    logger.info(f"Starting GCN pipeline for dataset: {dataset_name}")
    
    # Load configuration
    try:
        # No need for separate config loading - pipeline handles it
        config = {}
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
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model hyperparameters
    hyperparams = {
        # Standard parameters
        "embedding_dim": 128,
        "num_epoch": 100,         # Epochs for training the GCN encoder
        "learning_rate": 0.01,
        
        # GCN-specific parameters
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.5,
        "remove_self_loops": False, # From our GCNConv refactoring

        # Decoder parameters (standardized)
        "decoder_epochs": 100,      # Epochs for the final classifier
        "decoder_learning_rate": 0.01,
    }
    
    logger.info(f"Model hyperparameters: {hyperparams}")
    
    try:
        # Step 1: Load and preprocess graph data using the pipeline system (black box)
        logger.info("Step 1: Loading and preprocessing graph data...")
        start_time = time.time()
        
        # Use the preprocessing pipeline - completely abstracted data processing
        pipeline = Pipeline.from_config(f"{dataset_name}-example")
        processed_data = pipeline.run()
        
        # Convert to TemporalGraphData then to Graph object using proper abstraction
        temporal_graph = processed_data.to_temporal_graph()
        
        # Convert to final Graph object using the dedicated conversion function
        # NO MORE MANUAL DATA ASSEMBLY - this is the proper "black box" approach
        graph = build_graph_from_temporal(temporal_graph)
        
        logger.info(f"Data processing and conversion completed in {time.time() - start_time:.2f} seconds")
        
        # Step 2: Initialize GCN model
        logger.info("Step 2: Initializing GCN model...")
        start_time = time.time()
        
        model = GCNModel(
            device=device,
            hyperparams=hyperparams,
            epoch_evaluation_metric=roc_auc_score
        )
        
        model.setup(graph)
        logger.info(f"Model initialization took {time.time() - start_time:.2f} seconds")
        
        # Step 3: Train GCN model
        logger.info("Step 3: Training GCN model...")
        start_time = time.time()
        
        model.train()
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model...")
        
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
        logger.info(f"Model: GCN")
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