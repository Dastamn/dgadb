# Résultats des Expériences - Dataset Global-APT

## 📋 Résumé Exécutif

Ce document résume le travail effectué pour exécuter les algorithmes de détection d'anomalies sur le dataset **global-apt-simplified-dataset.json**.

## ✅ Ce qui a été fait

1. **Préparation du dataset** ✅
   - Conversion du JSON vers format Parquet
   - Mapping des adresses IP vers IDs numériques
   - Création des fichiers `edges.parquet` et `edge_labels.parquet`
   - Dataset prêt dans `data/global-apt/`

2. **Configuration** ✅
   - Création du fichier de configuration `configs/global-apt.yaml`
   - Configuration du split train/test/validation (70/15/15)

3. **Scripts d'exécution** ✅
   - `scripts/prepare_global_apt.py` : Préparation du dataset
   - `scripts/run_global_apt_experiments.py` : Exécution des algorithmes
   - `scripts/visualize_results.py` : Visualisation des résultats

4. **Documentation** ✅
   - README détaillé avec schéma explicatif dans `results/global-apt/README.md`
   - Ce document de synthèse

## 📊 Dataset

- **Fichier source** : `/Users/ie03-research/Desktop/global-apt-simplified-dataset.json`
- **Total** : 4,632 événements
- **Anomalies** : 1,544 (33.33%)
- **Normaux** : 3,088 (66.67%)
- **Nœuds** : 5 nœuds uniques

## 🔄 Pipeline Complet

```
JSON Dataset → Conversion Parquet → Split Temporel → Algorithmes → Métriques
```

Voir le schéma détaillé dans `results/global-apt/README.md`

## 🚀 Comment Utiliser

### 1. Préparer le dataset (déjà fait)

```bash
cd /Users/ie03-research/Desktop/dgadb-main
source venv/bin/activate
export BASE_PATH=$(pwd)
python scripts/prepare_global_apt.py
```

### 2. Exécuter les expériences

```bash
python scripts/run_global_apt_experiments.py
```

### 3. Visualiser les résultats

```bash
python scripts/visualize_results.py
```

Ou consulter directement les fichiers JSON dans `results/global-apt/`

## 📈 Statut des Algorithmes

| Algorithme | Statut | Notes |
|------------|--------|-------|
| **GeneralDYG** | ✅ En cours | Exécution en cours, peut prendre 5-15 minutes |
| **SLADE** | ⚠️ Dépendance manquante | Nécessite `torch_scatter` |
| **Node2Vec** | ⏸️ À implémenter | Nécessite preprocessing via pipeline |

## 📁 Fichiers Créés

```
dgadb-main/
├── data/global-apt/
│   ├── edges.parquet
│   ├── edge_labels.parquet
│   └── node_mapping.json
├── configs/
│   └── global-apt.yaml
├── scripts/
│   ├── prepare_global_apt.py
│   ├── run_global_apt_experiments.py
│   └── visualize_results.py
└── results/global-apt/
    ├── README.md (documentation complète avec schéma)
    └── results_*.json (résultats des expériences)
```

## 🔧 Dépendances

Toutes les dépendances principales sont installées. Pour SLADE, il faut :

```bash
pip install torch-scatter
```

## 📝 Prochaines Étapes

1. Attendre la fin de l'exécution de GeneralDYG
2. Installer `torch_scatter` pour exécuter SLADE
3. Implémenter le preprocessing pour Node2Vec si nécessaire
4. Analyser les résultats et comparer les performances

## 📚 Documentation

- **README principal** : `results/global-apt/README.md` (documentation complète avec schéma)
- **Résultats** : Fichiers JSON dans `results/global-apt/`

---

**Date** : 2026-02-04  
**Auteur** : Équipe DGADB
