# dgadb - Dynamic Graph Anomaly Detection Benchmark

**dgadb** est un framework de benchmark pour la détection d'anomalies dans les graphes dynamiques. Il implémente plusieurs algorithmes de détection d'anomalies et fournit une interface unifiée pour les exécuter sur vos datasets.

## 📋 Table des matières

- [Installation](#installation)
- [Format des données](#format-des-données)
- [Préparation de votre dataset](#préparation-de-votre-dataset)
- [Exécution des algorithmes](#exécution-des-algorithmes)
- [Algorithmes disponibles](#algorithmes-disponibles)
- [Configuration](#configuration)

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip
- virtualenv (recommandé)

### Étapes d'installation

```bash
# Cloner le repository
cd dgadb-main

# Définir la variable d'environnement BASE_PATH
export BASE_PATH=$(pwd)

# Créer un environnement virtuel
virtualenv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Format des données

Le framework attend vos données dans un format spécifique organisé en fichiers Parquet dans un dossier `data/<nom-du-dataset>/`.

### Structure du dossier dataset

```
data/
  └── <nom-du-dataset>/
      ├── edges.parquet              # OBLIGATOIRE
      ├── edge_labels.parquet        # Optionnel (pour datasets labellisés)
      ├── edge_features_num.parquet # Optionnel
      ├── edge_features_cat.parquet # Optionnel
      ├── edge_features_str.parquet # Optionnel
      ├── edge_types.parquet        # Optionnel
      ├── node_labels.parquet       # Optionnel
      ├── node_features_num.parquet # Optionnel
      ├── node_features_cat.parquet # Optionnel
      ├── node_features_str.parquet # Optionnel
      ├── node_types.parquet        # Optionnel
      └── node_timestamps.parquet   # Optionnel
```

### Format des fichiers

#### `edges.parquet` (OBLIGATOIRE)

Doit contenir les colonnes suivantes :
- `edge_id` : Identifiant unique de l'arête (entier)
- `src` : Identifiant du nœud source (entier)
- `tgt` : Identifiant du nœud destination (entier)
- `timestamp` : Timestamp de l'événement (entier ou float)

Exemple :
```python
import polars as pl

df_edges = pl.DataFrame({
    "edge_id": [0, 1, 2, 3],
    "src": [0, 1, 2, 0],
    "tgt": [1, 2, 0, 2],
    "timestamp": [1000, 2000, 3000, 4000]
})
df_edges.write_parquet("data/mon-dataset/edges.parquet")
```

#### `edge_labels.parquet` (Pour datasets labellisés)

Format **option 1** - Format simple (une colonne) :
- `edge_id` : Identifiant de l'arête
- `label` : Label binaire (0 = normal, 1 = anomalie)

Format **option 2** - Format pivot (pour plusieurs labels) :
- `edge_id` : Identifiant de l'arête
- `feature_id` : Identifiant du label
- `value` : Valeur du label

Exemple format simple :
```python
df_edge_labels = pl.DataFrame({
    "edge_id": [0, 1, 2, 3],
    "label": [0, 1, 0, 1]  # 0 = normal, 1 = anomalie
})
df_edge_labels.write_parquet("data/mon-dataset/edge_labels.parquet")
```

#### `edge_features_num.parquet` (Features numériques)

Format **option 1** - Format simple (colonnes directes) :
- `edge_id` : Identifiant de l'arête
- `f0`, `f1`, ... : Features numériques

Format **option 2** - Format pivot :
- `edge_id` : Identifiant de l'arête
- `feature_id` : Identifiant de la feature
- `value` : Valeur numérique

Exemple format simple :
```python
df_edge_features = pl.DataFrame({
    "edge_id": [0, 1, 2, 3],
    "f0": [1.5, 2.3, 0.8, 1.2],
    "f1": [0.1, 0.5, 0.3, 0.2]
})
df_edge_features.write_parquet("data/mon-dataset/edge_features_num.parquet")
```

## 📝 Préparation de votre dataset

### Étape 1 : Organiser vos données

Si vous avez un dataset d'événements labellisés, vous devez le convertir au format attendu :

```python
import polars as pl
import os

# Exemple : conversion depuis un CSV
df = pl.read_csv("votre_dataset.csv")

# Créer le dossier de destination
dataset_name = "mon-dataset"
os.makedirs(f"data/{dataset_name}", exist_ok=True)

# Créer le fichier edges.parquet
df_edges = df.select([
    pl.int_range(0, len(df)).alias("edge_id"),
    pl.col("source").alias("src"),
    pl.col("target").alias("tgt"),
    pl.col("timestamp")
])
df_edges.write_parquet(f"data/{dataset_name}/edges.parquet")

# Créer le fichier edge_labels.parquet (si vous avez des labels)
if "label" in df.columns:
    df_labels = pl.DataFrame({
        "edge_id": df_edges["edge_id"],
        "label": df["label"]
    })
    df_labels.write_parquet(f"data/{dataset_name}/edge_labels.parquet")
```

### Script de conversion complet

Créez un fichier `scripts/prepare_my_dataset.py` :

```python
#!/usr/bin/env python3
"""
Script pour préparer votre dataset d'événements labellisés
"""
import polars as pl
import os
import sys

def prepare_dataset(csv_path: str, dataset_name: str):
    """
    Convertit un CSV d'événements labellisés au format dgadb
    
    Format CSV attendu :
    - source, target, timestamp, label (optionnel)
    - ou toute autre colonne que vous pouvez mapper
    """
    # Lire le CSV
    df = pl.read_csv(csv_path)
    print(f"Dataset chargé : {len(df)} événements")
    print(f"Colonnes : {df.columns}")
    
    # Créer le dossier de destination
    base_path = os.environ.get("BASE_PATH", ".")
    dataset_dir = os.path.join(base_path, "data", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Mapper les colonnes (ajustez selon votre format)
    # Exemple si vos colonnes s'appellent différemment :
    column_mapping = {
        "source": "src",
        "target": "tgt", 
        "timestamp": "timestamp",
        "label": "label"
    }
    
    # Créer edges.parquet
    df_edges = df.select([
        pl.int_range(0, len(df)).alias("edge_id"),
        pl.col("source").alias("src"),  # Ajustez selon vos colonnes
        pl.col("target").alias("tgt"),
        pl.col("timestamp")
    ])
    
    # Trier par timestamp
    df_edges = df_edges.sort("timestamp")
    
    edges_path = os.path.join(dataset_dir, "edges.parquet")
    df_edges.write_parquet(edges_path)
    print(f"✓ Fichier edges créé : {edges_path}")
    
    # Créer edge_labels.parquet si labels présents
    if "label" in df.columns:
        df_labels = pl.DataFrame({
            "edge_id": df_edges["edge_id"],
            "label": df["label"].cast(pl.Int64)
        })
        labels_path = os.path.join(dataset_dir, "edge_labels.parquet")
        df_labels.write_parquet(labels_path)
        print(f"✓ Fichier labels créé : {labels_path}")
        print(f"  - Anomalies : {df_labels['label'].sum()}")
        print(f"  - Normaux : {len(df_labels) - df_labels['label'].sum()}")
    
    # Ajouter des features numériques si disponibles
    feature_cols = [col for col in df.columns 
                   if col not in ["source", "target", "timestamp", "label"] 
                   and df[col].dtype in [pl.Float64, pl.Int64]]
    
    if feature_cols:
        df_features = df_edges.select(["edge_id"]).with_columns([
            df[col].alias(f"f{i}") 
            for i, col in enumerate(feature_cols)
        ])
        # Convertir au format pivot si nécessaire
        features_path = os.path.join(dataset_dir, "edge_features_num.parquet")
        df_features.write_parquet(features_path)
        print(f"✓ Fichier features créé : {features_path}")
    
    print(f"\n✓ Dataset '{dataset_name}' préparé avec succès !")
    print(f"  Emplacement : {dataset_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_my_dataset.py <chemin_csv> <nom_dataset>")
        print("Exemple: python prepare_my_dataset.py data/events.csv mon-dataset")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    dataset_name = sys.argv[2]
    
    # Définir BASE_PATH si non défini
    if "BASE_PATH" not in os.environ:
        os.environ["BASE_PATH"] = os.getcwd()
    
    prepare_dataset(csv_path, dataset_name)
```

Utilisation :
```bash
export BASE_PATH=$(pwd)
python scripts/prepare_my_dataset.py votre_dataset.csv mon-dataset
```

### Étape 2 : Créer un fichier de configuration

Créez un fichier YAML dans le dossier `configs/` :

```yaml
# configs/mon-dataset.yaml
dataset:
  name: mon-dataset
  paths:
    raw: null
    structured: data/mon-dataset
    processed: null

pipeline:
  cache_dir: pipeline_cache/mon-dataset
  steps:
    - name: StructureNormalizer
      params:
        directionality: canonical
        reindex_nodes: True
        
    - name: TimestampNormalizer

    - name: TemporalSplitter
      params:
        train_ratio: 0.7
        val_ratio: 0.1

# Paramètres spécifiques aux algorithmes
train_ratio: 0.7
val_ratio: 0.1
window_size: 1000  # Pour certains algorithmes
anomaly_ratio: 0.05  # Pour certains algorithmes
```

## 🎯 Exécution des algorithmes

### Méthode générale

Tous les algorithmes suivent un pattern similaire. Assurez-vous d'avoir défini `BASE_PATH` :

```bash
export BASE_PATH=$(pwd)
```

### Algorithmes disponibles

#### 1. SLADE

```bash
python -m src.dgadb.pipeline.pipeline_SLADE
```

Ou en modifiant le script pour utiliser votre dataset :
```python
# Dans pipeline_SLADE.py, changer :
dataset = "mon-dataset"  # au lieu de "bitcoin-alpha"
```

#### 2. GeneralDYG

```bash
python -m src.dgadb.pipeline.pipeline_GeneralDYG
```

#### 3. SAD

```bash
python -m src.dgadb.pipeline.pipeline_SAD
```

Dans le script, modifiez :
```python
config_name = "mon-dataset"  # au lieu de "mooc"
```

#### 4. GAT (Graph Attention Network)

```bash
python -m src.dgadb.pipeline.pipeline_GAT
```

#### 5. GraphSAGE

```bash
python -m src.dgadb.pipeline.pipeline_GraphSAGE
```

#### 6. Node2Vec

```bash
python -m src.dgadb.pipeline.pipeline_Node2Vec
```

#### 7. NetWalk

```bash
python -m src.dgadb.pipeline.pipeline_NetWalk
```

#### 8. RustGraph

```bash
python -m src.dgadb.pipeline.pipeline_RustGraph
```

#### 9. StrGNN

```bash
python -m src.dgadb.pipeline.pipeline_StrGNN
```

#### 10. TADDY

```bash
python -m src.dgadb.pipeline.pipeline_TADDY
```

### Exemple complet : Exécuter SLADE sur votre dataset

1. **Préparer les données** (voir section précédente)

2. **Créer la configuration** `configs/mon-dataset.yaml` :
```yaml
train_ratio: 0.7
val_ratio: 0.1
window_size: 1000
```

3. **Modifier le pipeline** `src/dgadb/pipeline/pipeline_SLADE.py` :
```python
dataset = "mon-dataset"  # Ligne 18
```

4. **Exécuter** :
```bash
export BASE_PATH=$(pwd)
python -m src.dgadb.pipeline.pipeline_SLADE
```

## 🔧 Configuration

### Variables d'environnement

- `BASE_PATH` : Chemin racine du projet (obligatoire)
- `DATA_PATH` : Chemin vers les données (optionnel, certains scripts l'utilisent)

### Structure des fichiers de configuration

Les fichiers YAML dans `configs/` peuvent contenir :

```yaml
# Configuration du dataset
dataset:
  name: nom-du-dataset
  paths:
    structured: data/nom-du-dataset

# Configuration du pipeline de preprocessing
pipeline:
  cache_dir: pipeline_cache/nom-du-dataset
  steps:
    - name: StructureNormalizer
      params:
        directionality: canonical
        reindex_nodes: True
    - name: TimestampNormalizer
    - name: TemporalSplitter
      params:
        train_ratio: 0.7
        val_ratio: 0.1

# Paramètres pour les algorithmes
train_ratio: 0.7
val_ratio: 0.1
window_size: 1000
anomaly_ratio: 0.05
snapshot_size: 1000
```

## 📚 Algorithmes disponibles

| Algorithme | Description | Pipeline |
|------------|-------------|----------|
| **SLADE** | Self-supervised Learning for Anomaly Detection | `pipeline_SLADE.py` |
| **GeneralDYG** | General Dynamic Graph model | `pipeline_GeneralDYG.py` |
| **SAD** | Structural Anomaly Detection | `pipeline_SAD.py` |
| **GAT** | Graph Attention Network | `pipeline_GAT.py` |
| **GraphSAGE** | Graph Sample and Aggregate | `pipeline_GraphSAGE.py` |
| **Node2Vec** | Node embedding method | `pipeline_Node2Vec.py` |
| **NetWalk** | Network embedding for anomaly detection | `pipeline_NetWalk.py` |
| **RustGraph** | Rust-based graph processing | `pipeline_RustGraph.py` |
| **StrGNN** | Structural Graph Neural Network | `pipeline_StrGNN.py` |
| **TADDY** | Temporal Anomaly Detection | `pipeline_TADDY.py` |

## 💡 Conseils

1. **Pour les datasets labellisés** : Assurez-vous que `edge_labels.parquet` contient bien les labels correspondant à chaque `edge_id` dans `edges.parquet`.

2. **Normalisation des timestamps** : Les timestamps doivent être numériques. Si vous avez des dates, convertissez-les en timestamps Unix.

3. **Réindexation des nœuds** : Si vos nœuds ne sont pas numérotés de 0 à N-1, le pipeline peut les réindexer automatiquement avec `reindex_nodes: True`.

4. **GPU vs CPU** : La plupart des algorithmes détectent automatiquement la disponibilité du GPU. Pour forcer l'utilisation du CPU, modifiez les scripts pour utiliser `device = torch.device("cpu")`.

## 🐛 Dépannage

### Erreur : "Config file does not exist"
- Vérifiez que votre fichier de configuration existe dans `configs/<nom>.yaml`
- Vérifiez que `BASE_PATH` est correctement défini

### Erreur : "Required file not found: edges.parquet"
- Vérifiez que le fichier `edges.parquet` existe dans `data/<nom-du-dataset>/`
- Vérifiez que le nom du dataset dans la config correspond au nom du dossier

### Erreur : "CUDA out of memory"
- Réduisez la taille du batch ou utilisez le CPU
- Réduisez `window_size` ou `snapshot_size` dans la configuration

## 📄 Licence

Voir le fichier `LICENSE` pour plus d'informations.

## 🤝 Contribution

Ce projet est un benchmark pour la détection d'anomalies dans les graphes dynamiques. Pour contribuer, veuillez suivre les conventions de code existantes.
