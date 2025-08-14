#!/usr/bin/env python3
"""
GAT Pipeline for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This pipeline implements GAT-based anomaly detection using PyTorch Geometric.
It learns node embeddings through graph attention mechanisms and uses them for edge-level
anomaly detection via a downstream classifier.
"""

import logging
import numpy as np
import polars as pl
import sys
import time

import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.data.dataset import load_df
from src.dgadb.models.GAT.GAT_main import GATModel
from src.dgadb.preprocessing.anomaly_generation import AnomalyGenerator
from src.dgadb.preprocessing.snapshotting import assign_snapshots
from src.dgadb.preprocessing.splitting import generate_data_splits
from src.dgadb.utils import load_config

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
        # Step 1: Load dataset
        logger.info("Step 1: Loading dataset...")
        start_time = time.time()
        
        data_dict = load_df(dataset_name)
        df_edges = data_dict["edges"]
        logger.info(f"Loaded {len(df_edges)} edges")
        logger.info(f"Data loading took {time.time() - start_time:.2f} seconds")
        
        # Step 2: Generate train/test splits
        logger.info("Step 2: Generating train/test splits...")
        start_time = time.time()
        
        data_dict_with_splits = generate_data_splits(
            data_dict,
            train_ratio=config.get("train_ratio", 0.70),
            val_ratio=0.0  # No validation split for simplicity
        )
        df_edges = data_dict_with_splits["edges"]
        
        train_count = df_edges.filter(df_edges["train_mask"]).shape[0]
        test_count = df_edges.filter(df_edges["test_mask"]).shape[0]
        val_count = df_edges.filter(df_edges.get("val_mask", [False] * len(df_edges))).shape[0] if "val_mask" in df_edges.columns else 0
        
        logger.info(f"Train edges: {train_count}, Test edges: {test_count}, Val edges: {val_count}")
        logger.info(f"Data splitting took {time.time() - start_time:.2f} seconds")
        
        # Step 3: Create temporal snapshots
        logger.info("Step 3: Creating temporal snapshots...")
        start_time = time.time()
        
        data_dict = assign_snapshots(
            {"edges": df_edges},
            snapshot_size=config.get("snapshot_size", 1000),
            temporal_snapshots=False  # Use structural snapshots for GAT
        )
        
        df_edges = data_dict["edges"]
        logger.info(f"Created snapshots with window size: {config.get('snapshot_size', 1000)}")
        logger.info(f"Snapshotting took {time.time() - start_time:.2f} seconds")
        
        # Step 4: Inject anomalies into both train and test sets
        logger.info("Step 4: Injecting anomalies...")
        start_time = time.time()
        
        # Create dummy edge features for anomaly generation (GAT doesn't use edge features)
        edge_features = np.ones((len(df_edges), 1))  # Dummy features
        
        # First, inject anomalies into training set
        anomaly_gen = AnomalyGenerator(df_edges, edge_features)
        
        # Create a temporary train_mask column for anomaly injection
        df_edges_with_train_mask = df_edges.with_columns(
            df_edges["train_mask"].alias("test_mask")  # Temporarily use train_mask as test_mask
        )
        
        # Inject anomalies into training set
        anomaly_gen_train = AnomalyGenerator(df_edges_with_train_mask, edge_features)
        df_edges_train, _ = anomaly_gen_train._generate_anomalous_samples(
            anom_ratio=config.get("anomaly_ratio", 0.01) * 0.5,  # Use half the ratio for training
            anom_type="structural",
            use_val_split=False,  # This will use the "test_mask" which is actually train_mask
            temporal_window_size=config.get("temporal_window_size", 100)
        )
        
        # Now inject anomalies into test set with the original data
        df_edges, _ = anomaly_gen._generate_anomalous_samples(
            anom_ratio=config.get("anomaly_ratio", 0.01),
            anom_type="structural",  # Use structural anomalies for GAT
            use_val_split=False,
            temporal_window_size=config.get("temporal_window_size", 100)
        )
        
        # Combine the training anomalies with the main dataset
        # Get only the anomalous edges from training set
        train_anomalies = df_edges_train.filter((df_edges_train["label"] == 1) & (df_edges_train["test_mask"] == True))
        if len(train_anomalies) > 0:
            # Update the train_mask and test_mask for training anomalies
            train_anomalies = train_anomalies.with_columns([
                pl.lit(True).alias("train_mask"),
                pl.lit(False).alias("test_mask")
            ])
            # Add training anomalies to the main dataset
            df_edges = pl.concat([df_edges, train_anomalies], how="vertical").sort("timestamp")
        
        total_anomalies = len(df_edges.filter(df_edges["label"] == 1))
        total_edges = len(df_edges)
        actual_anomaly_ratio = total_anomalies / total_edges
        
        logger.info(f"Injected {total_anomalies} anomalies out of {total_edges} edges")
        logger.info(f"Actual anomaly ratio: {actual_anomaly_ratio:.4f}")
        logger.info(f"Anomaly injection took {time.time() - start_time:.2f} seconds")
        
        # Step 5: Initialize GAT model
        logger.info("Step 5: Initializing GAT model...")
        start_time = time.time()
        
        model = GATModel(
            device=device,
            hyperparams=hyperparams,
            epoch_evaluation_metric=roc_auc_score
        )
        
        model.setup(df_edges)
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
        logger.info(f"Total Edges: {total_edges}")
        logger.info(f"Anomaly Ratio: {actual_anomaly_ratio:.4f}")
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