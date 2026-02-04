#!/usr/bin/env python3
"""
Script pour visualiser les résultats des expériences
"""
import json
import sys
from pathlib import Path
import os

def print_results_table(results_file):
    """Affiche les résultats sous forme de tableau"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 80)
    print("RÉSULTATS DES EXPÉRIENCES - DATASET GLOBAL-APT")
    print("=" * 80)
    
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        print("-" * 80)
        
        if "error" in metrics:
            print(f"  ❌ Erreur: {metrics['error']}")
        elif "status" in metrics:
            print(f"  ⏸️  Statut: {metrics['status']}")
            if "reason" in metrics:
                print(f"     Raison: {metrics['reason']}")
        else:
            # Afficher les métriques
            if "roc_auc" in metrics and metrics["roc_auc"]:
                print(f"  📊 ROC-AUC: {metrics['roc_auc']:.4f}")
            
            if "precision" in metrics:
                print(f"  🎯 Precision: {metrics['precision']:.4f}")
            
            if "recall" in metrics:
                print(f"  🔍 Recall: {metrics['recall']:.4f}")
            
            if "f1" in metrics:
                print(f"  ⚖️  F1-Score: {metrics['f1']:.4f}")
            
            if "confusion_matrix" in metrics:
                cm = metrics['confusion_matrix']
                print(f"  📈 Matrice de confusion:")
                print(f"     TN={metrics.get('tn', 'N/A')}, FP={metrics.get('fp', 'N/A')}")
                print(f"     FN={metrics.get('fn', 'N/A')}, TP={metrics.get('tp', 'N/A')}")
            
            if "inference_time" in metrics and metrics["inference_time"]:
                print(f"  ⏱️  Temps d'inférence: {metrics['inference_time']:.2f}s")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    base_path = os.environ.get("BASE_PATH", ".")
    results_dir = Path(base_path) / "results" / "global-apt"
    
    # Trouver le dernier fichier de résultats
    result_files = sorted(results_dir.glob("results_*.json"), reverse=True)
    
    if not result_files:
        print("Aucun fichier de résultats trouvé.")
        sys.exit(1)
    
    latest_file = result_files[0]
    print(f"📄 Lecture du fichier: {latest_file}")
    print_results_table(latest_file)
