#!/usr/bin/env python3
"""
Script pour préparer le dataset global-apt au format dgadb
"""
import json
import polars as pl
import os
import sys

def prepare_global_apt_dataset(json_path: str, dataset_name: str = "global-apt"):
    """
    Convertit le JSON global-apt au format dgadb
    
    Args:
        json_path: Chemin vers le fichier JSON
        dataset_name: Nom du dataset dans data/
    """
    print(f"📖 Lecture du fichier JSON: {json_path}")
    
    # Lire le JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logs = data['logs']
    print(f"✓ {len(logs)} événements chargés")
    
    # Statistiques
    labels = [log['label'] for log in logs]
    anomalies = sum(labels)
    normaux = len(labels) - anomalies
    print(f"  - Anomalies: {anomalies} ({anomalies/len(labels)*100:.2f}%)")
    print(f"  - Normaux: {normaux} ({normaux/len(labels)*100:.2f}%)")
    
    # Créer le dossier de destination
    base_path = os.environ.get("BASE_PATH", ".")
    dataset_dir = os.path.join(base_path, "data", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"📁 Dossier de destination: {dataset_dir}")
    
    # Créer le DataFrame des edges
    df_edges = pl.DataFrame({
        "edge_id": list(range(len(logs))),
        "src": [log['src'] for log in logs],
        "tgt": [log['tgt'] for log in logs],
        "timestamp": [log['timestamp'] for log in logs]
    })
    
    # Convertir les IPs en IDs numériques (nécessaire pour les algorithmes)
    all_nodes = set(df_edges["src"].to_list() + df_edges["tgt"].to_list())
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    
    print(f"✓ {len(node_map)} nœuds uniques détectés")
    
    # Appliquer le mapping
    df_edges = df_edges.with_columns([
        pl.col("src").replace(node_map).cast(pl.Int64),
        pl.col("tgt").replace(node_map).cast(pl.Int64)
    ])
    
    # Trier par timestamp
    df_edges = df_edges.sort("timestamp")
    
    # Sauvegarder edges.parquet
    edges_path = os.path.join(dataset_dir, "edges.parquet")
    df_edges.write_parquet(edges_path)
    print(f"✓ Fichier edges créé: {edges_path}")
    print(f"  - Colonnes: {df_edges.columns}")
    print(f"  - Shape: {df_edges.shape}")
    
    # Créer edge_labels.parquet
    df_labels = pl.DataFrame({
        "edge_id": list(range(len(logs))),
        "label": [log['label'] for log in logs]
    })
    
    labels_path = os.path.join(dataset_dir, "edge_labels.parquet")
    df_labels.write_parquet(labels_path)
    print(f"✓ Fichier labels créé: {labels_path}")
    print(f"  - Distribution: {df_labels['label'].value_counts().sort('label')}")
    
    # Sauvegarder le mapping des nœuds pour référence
    mapping_path = os.path.join(dataset_dir, "node_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in node_map.items()}, f, indent=2)
    print(f"✓ Mapping des nœuds sauvegardé: {mapping_path}")
    
    print(f"\n✅ Dataset '{dataset_name}' préparé avec succès !")
    print(f"   Emplacement: {dataset_dir}")
    
    return dataset_dir

if __name__ == "__main__":
    json_path = "/Users/ie03-research/Desktop/global-apt-simplified-dataset.json"
    dataset_name = "global-apt"
    
    if "BASE_PATH" not in os.environ:
        os.environ["BASE_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"⚠️  BASE_PATH non défini, utilisation de: {os.environ['BASE_PATH']}")
    
    prepare_global_apt_dataset(json_path, dataset_name)
