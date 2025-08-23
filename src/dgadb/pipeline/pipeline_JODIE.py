#!/usr/bin/env python3
"""
JODIE Pipeline for Dynamic Graph Anomaly Detection Benchmark (DGADB)

This pipeline implements JODIE-based anomaly detection for temporal interaction networks.
JODIE learns dynamic embeddings for users and items through coupled RNNs and temporal projections.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks.
S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on 
Knowledge Discovery and Data Mining (KDD), 2019.
"""

import logging
import sys
import time

import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.JODIE.JODIE_main import JODIEModel
from src.dgadb.preprocessing.pipeline.pipeline import Pipeline
from src.dgadb.data.builder import build_graph_from_temporal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution for JODIE model."""
    
    # Configuration
    dataset_name = "bitcoin-alpha-example"  # Change this to your desired dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # JODIE-specific hyperparameters
    hyperparams = {
        # Standard parameters
        "embedding_dim": 128,
        "num_epoch": 5,  # Reduced for testing
        "learning_rate": 0.001,
        
        # Method-specific parameters
        "regularization_lambda": 0.001,  # For temporal smoothness loss
        
        # Decoder parameters (standardized)
        "decoder_epochs": 50,  # Reduced for testing
        "decoder_learning_rate": 0.01,
    }
    
    logger.info("="*80)
    logger.info("JODIE Pipeline for Dynamic Graph Anomaly Detection")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Hyperparameters: {hyperparams}")
    logger.info("="*80)
    
    try:
        # Step 1: Load and preprocess graph data using the pipeline system (black box)
        logger.info("Step 1: Loading and preprocessing graph data...")
        start_time = time.time()
        
        # Use the preprocessing pipeline - completely abstracted data processing
        pipeline = Pipeline.from_config(f"{dataset_name}")
        processed_data = pipeline.run()
        
        # Convert to TemporalGraphData then to Graph object using proper abstraction
        temporal_graph = processed_data.to_temporal_graph()
        
        # Convert to final Graph object using the dedicated conversion function
        # NO MORE MANUAL DATA ASSEMBLY - this is the proper "black box" approach
        graph = build_graph_from_temporal(temporal_graph)
        
        logger.info(f"Data processing and conversion completed in {time.time() - start_time:.2f} seconds")
        
        # Step 2: Initialize JODIE model
        logger.info("Step 2: Initializing JODIE model...")
        start_time = time.time()
        
        model = JODIEModel(
            device=device,
            hyperparams=hyperparams,
            epoch_evaluation_metric=roc_auc_score
        )
        
        model.setup(graph)
        logger.info(f"Model initialization took {time.time() - start_time:.2f} seconds")
        
        # Step 3: Train JODIE model
        logger.info("Step 3: Training JODIE model...")
        start_time = time.time()
        
        for epoch in range(hyperparams["num_epoch"]):
            epoch_start = time.time()
            model.train()
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % max(1, hyperparams["num_epoch"] // 10) == 0:
                logger.info(f"Epoch {epoch + 1}/{hyperparams['num_epoch']} completed in {epoch_time:.2f}s")
        
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
        logger.info(f"Model: JODIE")
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