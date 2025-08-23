#!/usr/bin/env python3
"""
EvolveGCN Pipeline for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This pipeline implements EvolveGCN-based anomaly detection for dynamic graphs.
"""

import logging
import sys
import time
import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.EvolveGCN.EvolveGCN_main import EvolveGCNModel
from src.dgadb.preprocessing.pipeline.pipeline import Pipeline
from src.dgadb.data.builder import build_graph_from_temporal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    
    dataset_name = "bitcoin-alpha"
    logger.info(f"Starting EvolveGCN pipeline for dataset: {dataset_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model hyperparameters - ULTRA LIGHTWEIGHT
    hyperparams = {
        "embedding_dim": 16,
        "num_epoch": 3,
        "learning_rate": 0.01,
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "remove_self_loops": False,
        "variant": "O",
        "decoder_epochs": 5,
        "decoder_learning_rate": 0.01,
    }
    logger.info(f"Model hyperparameters: {hyperparams}")

    try:
        # Step 1: Load and preprocess graph data using the black box pipeline
        logger.info("Step 1: Loading and preprocessing graph data...")
        start_time = time.time()
        pipeline = Pipeline.from_config(f"{dataset_name}-example")
        processed_data = pipeline.run()

        temporal_graph = processed_data.to_temporal_graph()
        graph = build_graph_from_temporal(temporal_graph)
        logger.info(f"Data processing completed in {time.time() - start_time:.2f} seconds")

        # Step 2: Initialize EvolveGCN model
        logger.info("Step 2: Initializing EvolveGCN model...")
        model = EvolveGCNModel(
            device=device,
            hyperparams=hyperparams,
            epoch_evaluation_metric=roc_auc_score
        )
        # Pass both temporal and final graph objects to setup
        model.setup(temporal_graph, graph)
        logger.info("Model initialization completed.")

        # Step 3: Train EvolveGCN model
        logger.info("Step 3: Training EvolveGCN model...")
        start_time = time.time()
        model.train()
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model...")
        test_preds, test_labels, test_inf_time = model.inference("test")
        test_auc = roc_auc_score(test_labels, test_preds)
        logger.info("Test Results:")
        logger.info(f"  - ROC-AUC Score: {test_auc:.4f}")
        logger.info(f"  - Inference Time: {test_inf_time:.2f} seconds")

        # Final Summary
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info(f"Dataset: {dataset_name}, Model: EvolveGCN")
        logger.info(f"Test AUC: {test_auc:.4f}, Training Time: {training_time:.2f}s")
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error("Pipeline failed with error:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()