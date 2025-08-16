#!/usr/bin/env python3
"""
Node2Vec Pipeline for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This pipeline implements Node2Vec-based anomaly detection using PyTorch Geometric.
It learns node embeddings through biased random walks and uses them for edge-level
anomaly detection via a downstream classifier.
"""

import logging
import sys
import time
import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.Node2Vec.Node2Vec_main import Node2VecModel
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
    
    logger.info(f"Starting Node2Vec pipeline for dataset: {dataset_name}")
    
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
            "anomaly_ratio": 0.01
        }
        logger.info(f"Using default config: {config}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model hyperparameters
    hyperparams = {
        # Node2Vec parameters
        "embedding_dim": 128,
        "walk_length": 20,
        "context_size": 10,
        "walks_per_node": 10,
        "p": 1.0,  # Return parameter
        "q": 1.0,  # In-out parameter
        "num_negative_samples": 1,
        "sparse": True,
        
        # Training parameters
        "num_epoch": 5,  # Reduced for testing
        "learning_rate": 0.01,
        "batch_size": 128,
        
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
        has_val = hasattr(graph, "e_val_mask")
        
        logger.info(f"Data loading and preprocessing took {time.time() - start_time:.2f} seconds")
        
        # Step 2: Initialize and setup model
        logger.info("Step 2: Initializing Node2Vec model...")
        start_time = time.time()
        
        model = Node2VecModel(
            device=device,
            hyperparams=hyperparams,
            epoch_evaluation_metric=roc_auc_score
        )
        
        model.setup(graph)
        logger.info(f"Model initialization took {time.time() - start_time:.2f} seconds")
        
        # Step 3: Train model
        logger.info("Step 3: Training Node2Vec model...")
        start_time = time.time()
        
        model.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model...")
        
        # Test set evaluation
        test_preds, test_labels, test_inf_time = model.inference("test")
        test_auc = roc_auc_score(test_labels, test_preds)
        
        logger.info(f"Test Results:")
        logger.info(f"  - ROC-AUC Score: {test_auc:.4f}")
        logger.info(f"  - Inference Time: {test_inf_time:.2f} seconds")
        logger.info(f"  - Test Edges: {len(test_labels)}")
        logger.info(f"  - Test Anomalies: {sum(test_labels)}")
        
        # Training set evaluation (for comparison)
        train_preds, train_labels, train_inf_time = model.inference("train")
        train_auc = roc_auc_score(train_labels, train_preds)
        
        logger.info(f"Train Results (for comparison):")
        logger.info(f"  - ROC-AUC Score: {train_auc:.4f}")
        logger.info(f"  - Inference Time: {train_inf_time:.2f} seconds")
        
        # Validation set evaluation (if available)
        val_auc = None
        if has_val:
            val_preds, val_labels, val_inf_time = model.inference("val")
            val_auc = roc_auc_score(val_labels, val_preds)
            
            logger.info(f"Validation Results:")
            logger.info(f"  - ROC-AUC Score: {val_auc:.4f}")
            logger.info(f"  - Inference Time: {val_inf_time:.2f} seconds")
        
        # Step 3: Summary
        logger.info("="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Model: Node2Vec")
        logger.info(f"Device: {device}")
        logger.info(f"Total Edges: {graph.num_edges}")
        logger.info(f"Training Time: {training_time:.2f} seconds")
        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}")
        if has_val:
            logger.info(f"Val AUC: {val_auc:.4f}")
        logger.info("="*60)
        
        return {
            "dataset": dataset_name,
            "model": "Node2Vec",
            "test_auc": test_auc,
            "train_auc": train_auc,
            "val_auc": val_auc if has_val else None,
            "training_time": training_time,
            "total_edges": graph.num_edges,
            "hyperparams": hyperparams
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Pipeline completed successfully!")
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)