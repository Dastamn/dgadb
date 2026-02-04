# Global-APT Dataset - Anomaly Detection Results

## 📊 Overview

This document presents the results of anomaly detection experiments performed on the **global-apt-simplified-dataset.json** dataset using the DGADB framework.

### Dataset Information

- **Source**: `/Users/ie03-research/Desktop/global-apt-simplified-dataset.json`
- **Total Events**: 4,632
- **Anomalies**: 1,544 (33.33%)
- **Normal Events**: 3,088 (66.67%)
- **Unique Nodes**: 5
- **Format**: Temporal graph with binary labels (0 = normal, 1 = anomaly)

### Train/Test/Validation Split

- **Train**: 70% (3,242 events)
- **Validation**: 15% (695 events)
- **Test**: 15% (695 events)

The split is performed **chronologically** (based on timestamps) to respect the temporal nature of the data.

## 📐 Graph Structure

The Global-APT dataset represents a small network with 5 nodes (IP addresses) and 4,632 temporal edges (connections).

### Graph Schema

```mermaid
graph LR
    N0["Node 0<br/>172.16.0.100<br/>Out: 384<br/>In: 385"]
    N1["Node 1<br/>172.16.0.50<br/>Out: 385<br/>In: 3475<br/>⭐ Hub"]
    N2["Node 2<br/>172.16.20.10<br/>Out: 1026<br/>In: 384"]
    N3["Node 3<br/>172.16.20.11<br/>Out: 1811<br/>In: 0<br/>🔥 Source"]
    N4["Node 4<br/>172.16.20.12<br/>Out: 1026<br/>In: 388"]
    
    N3 -->|1811 edges| N1
    N2 -->|1026 edges| N4
    N4 -->|1026 edges| N1
    N1 -->|385 edges| N0
    N0 -->|384 edges| N2
    N2 -->|1026 edges| N1
    
    style N1 fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    style N3 fill:#4ecdc4,stroke:#2d9cdb,stroke-width:2px
    style N0 fill:#ffe66d
    style N2 fill:#95e1d3
    style N4 fill:#95e1d3
```

### Graph Statistics

| Node ID | IP Address | Out-Degree | In-Degree | Role |
|---------|------------|------------|-----------|------|
| 0 | 172.16.0.100 | 384 | 385 | Balanced |
| 1 | 172.16.0.50 | 385 | **3,475** | **Hub (75% of traffic)** |
| 2 | 172.16.20.10 | 1,026 | 384 | Source |
| 3 | 172.16.20.11 | **1,811** | 0 | **Main Source** |
| 4 | 172.16.20.12 | 1,026 | 388 | Source |

### Key Observations

1. **Highly Centralized Network**: Node 1 (172.16.0.50) acts as a central hub, receiving 75% of all connections
2. **Unidirectional Flow**: Node 3 (172.16.20.11) only sends traffic (1,811 edges) but never receives
3. **Small Scale**: Only 5 nodes create a very dense graph with limited structural diversity
4. **Temporal Patterns**: 4,632 events over time create dynamic connection patterns

### Graph Characteristics Impact on Model Performance

The small, highly centralized graph structure contributes to the model's difficulty in detecting anomalies:

- **Limited Diversity**: With only 5 nodes, most connection patterns are frequent and "normal"
- **Hub Dominance**: The central hub (Node 1) receives most traffic, making it hard to distinguish anomalies
- **Sparse Anomaly Patterns**: Anomalies may be subtle structural or temporal deviations that are hard to detect in such a small graph
- **Frequency Bias**: The model learns that frequent patterns (to Node 1) are normal, missing rare but legitimate anomaly patterns

## 🔄 Pipeline Architecture

```mermaid
flowchart TD
    A[Global-APT JSON Dataset<br/>src, tgt, timestamp, label] --> B[Data Preparation]
    
    B --> B1[Convert JSON to Parquet]
    B --> B2[Map IPs to Numeric IDs]
    B --> B3[Sort by Timestamp]
    B --> B4[Create Files:<br/>edges.parquet<br/>edge_labels.parquet]
    
    B4 --> C[Temporal Split]
    
    C --> C1[Train Set<br/>70% - 3,242 events]
    C --> C2[Validation Set<br/>15% - 695 events]
    C --> C3[Test Set<br/>15% - 695 events]
    
    C1 --> D[Anomaly Detection Algorithms]
    C2 --> D
    C3 --> D
    
    D --> D1[SLADE<br/>TGN + Memory]
    D --> D2[GeneralDYG<br/>GNN + Temporal Attention]
    D --> D3[Node2Vec<br/>Embeddings + Random Walks]
    
    D1 --> E[Predictions<br/>Anomaly Scores]
    D2 --> E
    D3 --> E
    
    E --> F[Evaluation Metrics]
    
    F --> F1[ROC-AUC]
    F --> F2[Precision]
    F --> F3[Recall]
    F --> F4[F1-Score]
    F --> F5[Confusion Matrix]
    F --> F6[Inference Time]
    
    F1 --> G[Results JSON + README]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G
    F6 --> G
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#fce4ec
    style F fill:#e0f2f1
    style G fill:#fff9c4
```

## 📈 Algorithm Results

### GeneralDYG

**Status**: ✅ Completed

**Description**: 
- General Dynamic Graph model using Graph Neural Networks (GNN)
- Integrates temporal attention to capture temporal patterns
- Modular architecture with multiple processing layers

**Results**:
- **ROC-AUC**: 0.4942
- **Precision**: 1.0
- **Recall**: 0.0042 (0.42%)
- **F1-Score**: 0.0084

**Confusion Matrix**:
```
                Predicted
              Normal  Anomaly
Actual Normal   460      0
       Anomaly  235      1
```

**Analysis**:
- **True Positives (TP)**: 1
- **True Negatives (TN)**: 460
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 235

The model shows very high precision (1.0) but extremely low recall (0.42%), indicating it is very conservative in predicting anomalies. It correctly identifies normal events but misses most anomalies. The ROC-AUC of 0.4942 is close to random performance, suggesting the model needs further tuning or the dataset may require different preprocessing.

### Why Low Recall and "Everything Normal" Prediction?

The model's extremely low recall (0.42%) and tendency to predict everything as normal can be explained by several factors:

#### 1. **Graph Structure Characteristics**

The Global-APT dataset has a very small and highly connected graph structure:

- **Only 5 nodes**: This creates a very dense graph where most connections are frequent
- **Highly skewed degree distribution**: 
  - Node 1 (172.16.0.50) receives 75% of all connections (3,475 in-degree)
  - Node 3 (172.16.20.11) sends 39% of all connections (1,811 out-degree)
- **Limited structural diversity**: With only 5 nodes, the model has limited patterns to learn from

#### 2. **Class Imbalance in Training**

While the overall dataset has 33% anomalies, the temporal split may create imbalanced distributions:
- The model learns primarily from normal patterns during training
- Anomalies might be concentrated in specific time periods
- The model develops a bias toward predicting "normal" as the safe default

#### 3. **Lack of Discriminative Features**

The current setup uses only:
- **Structural features**: Source and target node IDs
- **Temporal features**: Timestamps
- **No edge features**: No additional attributes to distinguish anomalies

Without rich features, the model relies heavily on frequency patterns, which favor the majority class (normal events).

#### 4. **Conservative Learning Strategy**

The model's learning process:
- Minimizes false positives (high precision = 1.0)
- But at the cost of missing true anomalies (low recall = 0.42%)
- This suggests the loss function or threshold favors precision over recall
- The model learns that predicting "normal" is safer than risking false alarms

#### 5. **Small Dataset Size**

With only 4,632 events:
- Limited training data (3,242 events)
- Small test set (695 events) with only 236 anomalies expected
- The model may overfit to normal patterns
- Insufficient examples of anomaly patterns to learn from

#### Recommendations to Improve Recall:

1. **Adjust Classification Threshold**: Lower the decision threshold to increase recall
2. **Class Weighting**: Apply higher weights to anomaly class in loss function
3. **Feature Engineering**: Add node/edge features (e.g., connection frequency, time patterns)
4. **Different Loss Function**: Use F1-loss or recall-focused loss instead of standard cross-entropy
5. **Oversampling**: Use SMOTE or similar techniques to balance anomaly examples
6. **Ensemble Methods**: Combine multiple models with different thresholds

**Hyperparameters**:
- Batch size: 32
- Learning rate: 0.001
- Input dimension: 64
- Hidden dimension: 128
- Number of heads: 4
- Number of layers: 6
- Dropout: 0.3

### SLADE

**Status**: ⚠️ Dependency Missing

**Description**:
- Self-supervised Learning for Anomaly Detection
- Uses Temporal Graph Network (TGN) with memory
- Attention mechanism for temporal relationships

**Error**: `No module named 'torch_scatter'`

**Solution**: Install required dependency:
```bash
pip install torch-scatter
```

### Node2Vec

**Status**: ⏸️ Requires Pipeline Preprocessing

**Description**:
- Node embedding algorithm based on random walks
- Generates vector representations of nodes
- Uses downstream classifier for anomaly detection

**Note**: Requires additional preprocessing via the pipeline framework.

## 📊 Performance Summary

| Algorithm | ROC-AUC | Precision | Recall | F1-Score | Status |
|-----------|---------|-----------|--------|----------|--------|
| **GeneralDYG** | 0.4942 | 1.0 | 0.0042 | 0.0084 | ✅ Completed |
| **SLADE** | - | - | - | - | ⚠️ Dependency Missing |
| **Node2Vec** | - | - | - | - | ⏸️ Requires Preprocessing |

## 📁 File Structure

```
dgadb-main/
├── data/
│   └── global-apt/
│       ├── edges.parquet              # Edges with timestamps
│       ├── edge_labels.parquet        # Binary labels
│       └── node_mapping.json          # IP → ID mapping
├── configs/
│   └── global-apt.yaml                # Dataset configuration
├── scripts/
│   ├── prepare_global_apt.py          # Preparation script
│   ├── run_global_apt_experiments.py  # Execution script
│   └── visualize_results.py          # Visualization script
└── results/
    └── global-apt/
        ├── README.md                  # This file
        └── results_*.json            # Detailed results
```

## 🚀 How to Run

### 1. Prepare Dataset

```bash
cd /Users/ie03-research/Desktop/dgadb-main
source venv/bin/activate
export BASE_PATH=$(pwd)
python scripts/prepare_global_apt.py
```

### 2. Run Experiments

```bash
python scripts/run_global_apt_experiments.py
```

### 3. View Results

```bash
python scripts/visualize_results.py
```

Or directly view JSON files in `results/global-apt/results_*.json`

## 📊 Metrics Explanation

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **Range**: 0 to 1
- **Interpretation**:
  - 1.0 = Perfect classification
  - 0.5 = Random performance
  - > 0.7 = Good performance
  - > 0.9 = Excellent performance

### Precision
- **Formula**: TP / (TP + FP)
- **Interpretation**: Proportion of positive predictions that are correct
- **Important for**: Reducing false positives

### Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Interpretation**: Proportion of actual anomalies detected
- **Important for**: Not missing anomalies

### F1-Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretation**: Harmonic mean of Precision and Recall
- **Useful for**: Balancing Precision and Recall

### Confusion Matrix
```
                Predicted
              Normal  Anomaly
Actual Normal    TN      FP
       Anomaly   FN      TP
```

- **TP (True Positive)**: Anomalies correctly detected
- **TN (True Negative)**: Normal events correctly classified
- **FP (False Positive)**: False alarms (normal classified as anomaly)
- **FN (False Negative)**: Missed anomalies

## 🔧 Dependencies and Installation

### Required Dependencies

```bash
pip install torch torch-geometric polars pandas numpy scikit-learn
pip install matplotlib pyarrow tqdm pyyaml
```

### Optional Dependencies (for specific algorithms)

```bash
# For SLADE
pip install torch-scatter torch-cluster

# For other algorithms
pip install networkx
```

## 📝 Technical Notes

1. **Temporal Split**: The split is performed chronologically to respect the temporal nature of the data. Test data is always posterior to training data.

2. **Timestamp Normalization**: Timestamps are normalized to facilitate processing by models.

3. **Node Mapping**: IP addresses are converted to numeric IDs for computational efficiency.

4. **Memory Management**: Algorithms use temporal windows (window_size) to manage memory.

## 🐛 Known Issues and Solutions

### Error: "No module named 'torch_scatter'"
**Solution**: 
```bash
pip install torch-scatter
# Or for macOS with Apple Silicon:
pip install torch-scatter --index-url https://download.pytorch.org/whl/cpu
```

### Error: "No module named 'pyarrow'"
**Solution**:
```bash
pip install pyarrow
```

### Error: "No module named 'matplotlib'"
**Solution**:
```bash
pip install matplotlib
```

## 🔍 Analysis and Recommendations

### Current Results Analysis

The GeneralDYG model shows:
- **High Precision (1.0)**: No false positives - when it predicts an anomaly, it's correct
- **Very Low Recall (0.42%)**: Misses 99.58% of actual anomalies
- **Low ROC-AUC (0.4942)**: Close to random performance

### Recommendations

1. **Model Tuning**: Adjust hyperparameters, especially:
   - Learning rate
   - Number of layers
   - Hidden dimensions
   - Dropout rate

2. **Feature Engineering**: Consider adding:
   - Node features
   - Edge features
   - Temporal features

3. **Class Imbalance**: The dataset has 33% anomalies, but the model is very conservative. Consider:
   - Class weighting
   - Different loss functions
   - Resampling techniques

4. **Try Other Algorithms**: 
   - Install dependencies for SLADE
   - Implement Node2Vec preprocessing
   - Test other algorithms in the framework

## 📚 References

- **SLADE**: Self-supervised Learning for Anomaly Detection in Dynamic Graphs
- **GeneralDYG**: General Dynamic Graph Neural Networks
- **Node2Vec**: Scalable Feature Learning for Networks

## 🔄 Updates

Last updated: 2026-02-04

To update results, re-run:
```bash
python scripts/run_global_apt_experiments.py
```

---

**Author**: DGADB Team  
**Date**: February 2026
