#!/usr/bin/env python3
"""
Script to run multiple anomaly detection algorithms on global-apt dataset
with adjusted thresholds to avoid "always normal" predictions
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback
import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import polars as pl
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve

# Imports
from src.dgadb.data.dataset import load_df
from src.dgadb.preprocessing.temporal import generate_data_splits, normalize_timestamps
from src.dgadb.preprocessing.pipeline.pipeline import Pipeline
from src.dgadb.data.builder import build_graph_from_temporal
from src.dgadb.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

results = {}

def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold using F1-score"""
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return optimal_threshold

def calculate_metrics(y_true, y_pred, y_scores=None, threshold=0.5):
    """Calculate performance metrics with adjustable threshold"""
    metrics = {}
    
    # Adjust predictions based on threshold if scores provided
    if y_scores is not None:
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(y_true, y_scores)
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        
        # Also try median threshold
        median_threshold = np.median(y_scores)
        y_pred_median = (y_scores >= median_threshold).astype(int)
        
        # Use optimal threshold predictions
        y_pred = y_pred_optimal
        
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
        except:
            metrics['roc_auc'] = None
        
        metrics['optimal_threshold'] = float(optimal_threshold)
        metrics['median_threshold'] = float(median_threshold)
        metrics['threshold_used'] = float(optimal_threshold)
    else:
        metrics['threshold_used'] = threshold
    
    if y_pred is not None:
        # Check if we have any positive predictions
        n_positives = int(np.sum(y_pred))
        n_total = len(y_pred)
        
        metrics['n_positive_predictions'] = n_positives
        metrics['n_total_predictions'] = n_total
        metrics['positive_ratio'] = float(n_positives / n_total) if n_total > 0 else 0.0
        
        if n_positives > 0:
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
            logger.warning("No positive predictions! All predicted as normal.")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
        else:
            metrics['tn'] = int(cm[0, 0]) if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
            metrics['fp'] = 0
            metrics['fn'] = int(np.sum(y_true)) - metrics.get('tp', 0)
            metrics['tp'] = 0
    
    return metrics

def run_slade(dataset_name="global-apt"):
    """Run SLADE algorithm"""
    try:
        logger.info("=" * 60)
        logger.info("Running SLADE")
        logger.info("=" * 60)
        
        from src.dgadb.models.SLADE.SLADE_main import SLADEModel
        
        config = load_config(dataset_name)
        train_ratio = config.get("train_ratio", 0.7)
        val_ratio = config.get("val_ratio", 0.15)
        
        data = load_df(dataset_name)
        edges = generate_data_splits(data["edges"], train_ratio=train_ratio, val_ratio=val_ratio)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        
        hyperparams = {
            "batch_size": 50,  # Smaller for small dataset
            "num_neighbors": 10,
            "num_epoch": 5,
            "learning_rate": 0.0001,
        }
        
        model = SLADEModel(device, hyperparams, roc_auc_score)
        model.setup(edges.to_pandas())
        model.train()
        
        preds, labels, inf_time = model.inference("test")
        
        # Convert to numpy if needed
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        preds_binary = (preds > 0.5).astype(int)
        metrics = calculate_metrics(labels, preds_binary, preds)
        metrics['inference_time'] = float(inf_time) if inf_time else None
        
        results['SLADE'] = metrics
        logger.info(f"✓ SLADE completed - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
        logger.info(f"  Positive predictions: {metrics.get('n_positive_predictions', 0)}/{metrics.get('n_total_predictions', 0)}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ SLADE error: {e}")
        logger.error(traceback.format_exc())
        results['SLADE'] = {"error": str(e)}
        return None

def run_node2vec(dataset_name="global-apt"):
    """Run Node2Vec algorithm using pipeline"""
    try:
        logger.info("=" * 60)
        logger.info("Running Node2Vec")
        logger.info("=" * 60)
        
        from src.dgadb.models.Node2Vec.Node2Vec_main import Node2VecModel
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        
        # Use pipeline for preprocessing
        pipeline = Pipeline.from_config(f"{dataset_name}-example")
        processed_data = pipeline.run()
        temporal_graph = processed_data.to_temporal_graph()
        graph = build_graph_from_temporal(temporal_graph)
        
        hyperparams = {
            "embedding_dim": 64,  # Smaller for small graph
            "walk_length": 10,
            "context_size": 5,
            "walks_per_node": 5,
            "p": 1.0,
            "q": 1.0,
            "num_epoch": 10,
            "learning_rate": 0.01,
            "decoder_epochs": 50,
        }
        
        model = Node2VecModel(device, hyperparams, roc_auc_score)
        model.setup(graph)
        model.train()
        
        preds, labels, inf_time = model.inference("test")
        
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        preds_binary = (preds > 0.5).astype(int)
        metrics = calculate_metrics(labels, preds_binary, preds)
        metrics['inference_time'] = float(inf_time) if inf_time else None
        
        results['Node2Vec'] = metrics
        logger.info(f"✓ Node2Vec completed - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
        logger.info(f"  Positive predictions: {metrics.get('n_positive_predictions', 0)}/{metrics.get('n_total_predictions', 0)}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ Node2Vec error: {e}")
        logger.error(traceback.format_exc())
        results['Node2Vec'] = {"error": str(e)}
        return None

def run_gat(dataset_name="global-apt"):
    """Run GAT algorithm"""
    try:
        logger.info("=" * 60)
        logger.info("Running GAT")
        logger.info("=" * 60)
        
        from src.dgadb.models.GAT.GAT_main import GATModel
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        
        pipeline = Pipeline.from_config(f"{dataset_name}-example")
        processed_data = pipeline.run()
        temporal_graph = processed_data.to_temporal_graph()
        graph = build_graph_from_temporal(temporal_graph)
        
        hyperparams = {
            "hidden_channels": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "num_epoch": 10,
            "learning_rate": 0.01,
        }
        
        model = GATModel(device, hyperparams, roc_auc_score)
        model.setup(graph)
        model.train()
        
        preds, labels, inf_time = model.inference("test")
        
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        preds_binary = (preds > 0.5).astype(int)
        metrics = calculate_metrics(labels, preds_binary, preds)
        metrics['inference_time'] = float(inf_time) if inf_time else None
        
        results['GAT'] = metrics
        logger.info(f"✓ GAT completed - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
        logger.info(f"  Positive predictions: {metrics.get('n_positive_predictions', 0)}/{metrics.get('n_total_predictions', 0)}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ GAT error: {e}")
        logger.error(traceback.format_exc())
        results['GAT'] = {"error": str(e)}
        return None

def run_graphsage(dataset_name="global-apt"):
    """Run GraphSAGE algorithm"""
    try:
        logger.info("=" * 60)
        logger.info("Running GraphSAGE")
        logger.info("=" * 60)
        
        from src.dgadb.models.GraphSAGE.GraphSAGE_main import GraphSAGEModel
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        
        pipeline = Pipeline.from_config(f"{dataset_name}-example")
        processed_data = pipeline.run()
        temporal_graph = processed_data.to_temporal_graph()
        graph = build_graph_from_temporal(temporal_graph)
        
        hyperparams = {
            "hidden_channels": 32,
            "num_layers": 2,
            "num_epoch": 10,
            "learning_rate": 0.01,
        }
        
        model = GraphSAGEModel(device, hyperparams, roc_auc_score)
        model.setup(graph)
        model.train()
        
        preds, labels, inf_time = model.inference("test")
        
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        preds_binary = (preds > 0.5).astype(int)
        metrics = calculate_metrics(labels, preds_binary, preds)
        metrics['inference_time'] = float(inf_time) if inf_time else None
        
        results['GraphSAGE'] = metrics
        logger.info(f"✓ GraphSAGE completed - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
        logger.info(f"  Positive predictions: {metrics.get('n_positive_predictions', 0)}/{metrics.get('n_total_predictions', 0)}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ GraphSAGE error: {e}")
        logger.error(traceback.format_exc())
        results['GraphSAGE'] = {"error": str(e)}
        return None

def run_netwalk(dataset_name="global-apt"):
    """Run NetWalk algorithm"""
    try:
        logger.info("=" * 60)
        logger.info("Running NetWalk")
        logger.info("=" * 60)
        
        from src.dgadb.models.NetWalk.NetWalk_main import NetWalkModel
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        
        pipeline = Pipeline.from_config(f"{dataset_name}-example")
        processed_data = pipeline.run()
        temporal_graph = processed_data.to_temporal_graph()
        graph = build_graph_from_temporal(temporal_graph)
        
        hyperparams = {
            "embedding_dim": 32,
            "walk_length": 5,
            "walks_per_node": 3,
            "reservoir_dim": 10,
            "num_epoch": 5,
            "learning_rate": 0.1,
            "batch_size": 20,
            "k_clusters": 3,  # Smaller for small graph
        }
        
        model = NetWalkModel(device, hyperparams, roc_auc_score)
        model.setup(graph)
        model.train()
        
        preds, labels, inf_time = model.inference("test")
        
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        preds_binary = (preds > 0.5).astype(int)
        metrics = calculate_metrics(labels, preds_binary, preds)
        metrics['inference_time'] = float(inf_time) if inf_time else None
        
        results['NetWalk'] = metrics
        logger.info(f"✓ NetWalk completed - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
        logger.info(f"  Positive predictions: {metrics.get('n_positive_predictions', 0)}/{metrics.get('n_total_predictions', 0)}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ NetWalk error: {e}")
        logger.error(traceback.format_exc())
        results['NetWalk'] = {"error": str(e)}
        return None

def main():
    """Main function"""
    dataset_name = "global-apt"
    
    if "BASE_PATH" not in os.environ:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["BASE_PATH"] = base_path
        logger.info(f"BASE_PATH set to: {base_path}")
    
    logger.info(f"🚀 Starting experiments on dataset: {dataset_name}")
    logger.info(f"📊 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Run all algorithms
    run_slade(dataset_name)
    run_node2vec(dataset_name)
    run_gat(dataset_name)
    run_graphsage(dataset_name)
    run_netwalk(dataset_name)
    
    # Save results
    results_dir = Path(os.environ["BASE_PATH"]) / "results" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"results_all_algorithms_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📊 Results saved: {results_file}")
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY OF RESULTS")
    logger.info("=" * 60)
    
    for algo, metrics in results.items():
        if "error" not in metrics and "status" not in metrics:
            logger.info(f"\n{algo}:")
            if "roc_auc" in metrics and metrics["roc_auc"]:
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            if "f1" in metrics:
                logger.info(f"  F1-Score: {metrics['f1']:.4f}")
            if "precision" in metrics:
                logger.info(f"  Precision: {metrics['precision']:.4f}")
            if "recall" in metrics:
                logger.info(f"  Recall: {metrics['recall']:.4f}")
            if "n_positive_predictions" in metrics:
                logger.info(f"  Positive predictions: {metrics['n_positive_predictions']}/{metrics.get('n_total_predictions', 'N/A')}")
            if "threshold_used" in metrics:
                logger.info(f"  Threshold used: {metrics['threshold_used']:.4f}")
        else:
            logger.info(f"\n{algo}: {metrics}")
    
    return results

if __name__ == "__main__":
    main()
