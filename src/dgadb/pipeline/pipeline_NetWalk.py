#!/usr/bin/env python3
"""
NetWalk Pipeline for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This pipeline implements NetWalk-based anomaly detection using TensorFlow.
It learns node embeddings through clique embedding with deep autoencoder and uses them for edge-level
anomaly detection via streaming k-means clustering.
"""

import logging
import sys
import time
import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.NetWalk.NetWalk_main import NetWalkModel
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
    
    logger.info(f"Starting NetWalk pipeline for dataset: {dataset_name}")
    
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
        # NetWalk parameters
        "embedding_dim": 64,
        "walk_length": 3,
        "walks_per_node": 5,
        "reservoir_dim": 10,
        
        # Training parameters
        "num_epoch": 5,  # Reduced for testing
        "learning_rate": 0.1,
        "batch_size": 20,
        
        # Autoencoder parameters
        "gamma": 340.0,
        "lamb": 0.0017,
        "beta": 1.0,
        "rho": 0.5,
        
        # Clustering parameters
        "k_clusters": 5,
        "alpha": 0.5,
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
        logger.info("Step 2: Initializing NetWalk model...")
        start_time = time.time()
        
        model = NetWalkModel(
            device=device,
            hyperparams=hyperparams,
            epoch_evaluation_metric=roc_auc_score
        )
        
        model.setup(graph)
        logger.info(f"Model initialization took {time.time() - start_time:.2f} seconds")
        
        # Step 3: Train model
        logger.info("Step 3: Training NetWalk model...")
        start_time = time.time()
        
        model.initial_train()
        
        training_time = time.time() - start_time
        logger.info(f"Initial training completed in {training_time:.2f} seconds")
        
        # Step 4: Simulate streaming evaluation on the test set
        logger.info("Step 4: Simulating streaming evaluation on the test set...")

        # Let's assume the test data arrives in chunks (snapshots)
        test_edges = graph.e_pairs[:, graph.e_test_mask]
        test_labels = graph.e_label[graph.e_test_mask]

        snapshot_size = 100 # Example chunk size
        num_snapshots = (test_edges.shape[1] + snapshot_size - 1) // snapshot_size

        all_preds, all_labels = [], []
        total_inf_time = 0

        for i in range(num_snapshots):
            start_idx = i * snapshot_size
            end_idx = min((i + 1) * snapshot_size, test_edges.shape[1])
            
            edge_chunk = test_edges[:, start_idx:end_idx]
            label_chunk = test_labels[start_idx:end_idx]
            
            if edge_chunk.shape[1] == 0:
                continue
                
            logger.info(f"Processing snapshot {i+1}/{num_snapshots} with {edge_chunk.shape[1]} edges...")
            
            # This is the core streaming call
            preds, labels, inf_time = model.update_and_infer(edge_chunk, label_chunk)
            
            all_preds.append(preds)
            all_labels.append(labels)
            total_inf_time += inf_time

        # Consolidate results
        final_preds = [p for sublist in all_preds for p in sublist]
        final_labels = [l for sublist in all_labels for l in sublist]

        test_auc = roc_auc_score(final_labels, final_preds)

        logger.info(f"Test Results (from streaming evaluation):")
        logger.info(f"  - ROC-AUC Score: {test_auc:.4f}")
        logger.info(f"  - Total Inference Time: {total_inf_time:.2f} seconds")
        
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
        
        # Step 5: Summary
        logger.info("="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Model: NetWalk")
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
            "model": "NetWalk",
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