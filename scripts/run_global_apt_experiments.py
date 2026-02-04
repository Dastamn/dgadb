#!/usr/bin/env python3
"""
Script pour exécuter les algorithmes de détection d'anomalies sur le dataset global-apt
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import polars as pl
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Imports des algorithmes
from src.dgadb.data.dataset import load_df
from src.dgadb.preprocessing.temporal import generate_data_splits, normalize_timestamps
from src.dgadb.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Résultats globaux
results = {}

def calculate_metrics(y_true, y_pred, y_scores=None):
    """Calcule les métriques de performance"""
    metrics = {}
    
    # Métriques basées sur les scores (si disponibles)
    if y_scores is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
        except:
            metrics['roc_auc'] = None
    
    # Métriques basées sur les prédictions binaires
    if y_pred is not None:
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    return metrics

def run_slade(dataset_name="global-apt"):
    """Exécute l'algorithme SLADE"""
    try:
        logger.info("=" * 60)
        logger.info("Exécution de SLADE")
        logger.info("=" * 60)
        
        from src.dgadb.models.SLADE.SLADE_main import SLADEModel
        
        config = load_config(dataset_name)
        window_size = config.get("window_size", 500)
        train_ratio = config.get("train_ratio", 0.7)
        val_ratio = config.get("val_ratio", 0.15)
        
        # Charger les données
        data = load_df(dataset_name)
        edges = generate_data_splits(data["edges"], train_ratio=train_ratio, val_ratio=val_ratio)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        
        hyperparams = {}
        model = SLADEModel(device, hyperparams, roc_auc_score)
        model.setup(edges.to_pandas())
        model.train()
        
        # Inference sur test
        preds, labels, inf_time = model.inference("test")
        
        # Convertir en binaire pour certaines métriques
        preds_binary = (preds > 0.5).astype(int) if isinstance(preds, np.ndarray) else (preds > 0.5).int()
        
        metrics = calculate_metrics(labels, preds_binary, preds)
        metrics['inference_time'] = float(inf_time) if inf_time else None
        
        results['SLADE'] = metrics
        logger.info(f"✓ SLADE terminé - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ Erreur SLADE: {e}")
        logger.error(traceback.format_exc())
        results['SLADE'] = {"error": str(e)}
        return None

def run_generaldyg(dataset_name="global-apt"):
    """Exécute l'algorithme GeneralDYG"""
    try:
        logger.info("=" * 60)
        logger.info("Exécution de GeneralDYG")
        logger.info("=" * 60)
        
        from src.dgadb.models.GeneralDYG.GeneralDYG_main import GeneralDYGModel
        
        config = load_config(dataset_name)
        train_ratio = config.get("train_ratio", 0.7)
        val_ratio = config.get("val_ratio", 0.15)
        anomaly_ratio = config.get("anomaly_ratio", 0.33)
        
        meta_dict = {
            "dataset_name": dataset_name,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "anomaly_ratio": anomaly_ratio,
        }
        
        data = load_df(dataset_name)
        edges = generate_data_splits(data["edges"], train_ratio=train_ratio, val_ratio=val_ratio)
        edges = normalize_timestamps(edges)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        
        hyperparams = {}
        model = GeneralDYGModel(device, meta_dict, hyperparams, roc_auc_score)
        model.setup(edges)
        model.train()
        
        preds, labels, inf_time = model.inference("test")
        
        preds_binary = (preds > 0.5).astype(int) if isinstance(preds, np.ndarray) else (preds > 0.5).int()
        
        metrics = calculate_metrics(labels, preds_binary, preds)
        metrics['inference_time'] = float(inf_time) if inf_time else None
        
        results['GeneralDYG'] = metrics
        logger.info(f"✓ GeneralDYG terminé - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ Erreur GeneralDYG: {e}")
        logger.error(traceback.format_exc())
        results['GeneralDYG'] = {"error": str(e)}
        return None

def run_node2vec(dataset_name="global-apt"):
    """Exécute l'algorithme Node2Vec"""
    try:
        logger.info("=" * 60)
        logger.info("Exécution de Node2Vec")
        logger.info("=" * 60)
        
        # Node2Vec nécessite un preprocessing spécial via le pipeline
        logger.info("Node2Vec nécessite un preprocessing via pipeline - skip pour l'instant")
        results['Node2Vec'] = {"status": "skipped", "reason": "Nécessite preprocessing pipeline"}
        return None
        
    except Exception as e:
        logger.error(f"✗ Erreur Node2Vec: {e}")
        results['Node2Vec'] = {"error": str(e)}
        return None

def main():
    """Fonction principale"""
    dataset_name = "global-apt"
    
    # Vérifier que BASE_PATH est défini
    if "BASE_PATH" not in os.environ:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["BASE_PATH"] = base_path
        logger.info(f"BASE_PATH défini à: {base_path}")
    
    logger.info(f"🚀 Démarrage des expériences sur le dataset: {dataset_name}")
    logger.info(f"📊 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Statistiques du dataset
    try:
        data = load_df(dataset_name)
        logger.info(f"✓ Dataset chargé: {len(data['edges'])} événements")
        if 'edge_labels' in data:
            labels = data['edges'].join(data['edge_labels'], on='edge_id', how='left')['label']
            logger.info(f"  - Anomalies: {labels.sum()}, Normaux: {len(labels) - labels.sum()}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return
    
    # Exécuter les algorithmes
    run_slade(dataset_name)
    run_generaldyg(dataset_name)
    run_node2vec(dataset_name)
    
    # Sauvegarder les résultats
    results_dir = Path(os.environ["BASE_PATH"]) / "results" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📊 Résultats sauvegardés: {results_file}")
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ DES RÉSULTATS")
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
        else:
            logger.info(f"\n{algo}: {metrics}")
    
    return results

if __name__ == "__main__":
    main()
