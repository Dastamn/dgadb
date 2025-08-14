# DGADB Models Directory Documentation

This document provides a comprehensive guide to the models directory structure, patterns, and guidelines for adding new anomaly detection models to the DGADB framework.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Model Architecture Patterns](#model-architecture-patterns)
3. [Existing Models Overview](#existing-models-overview)
4. [Model Interface Requirements](#model-interface-requirements)
5. [Pipeline Integration](#pipeline-integration)
6. [Adding New Models](#adding-new-models)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)

## Directory Structure

The models directory follows a hierarchical structure where each model has its own subdirectory containing all necessary components:

```
src/dgadb/models/
├── __init__.py                 # Empty - models are imported directly
├── SLADE/                      # Temporal Graph Neural Network model
│   ├── __init__.py
│   ├── SLADE_main.py          # Main model wrapper class
│   ├── SLADE_TGN.py           # Core TGN implementation
│   ├── temporal_attention_SLADE.py
│   ├── time_encoding.py
│   ├── evaluation/            # Model-specific evaluation utilities
│   │   ├── __init__.py
│   │   └── evaluation.py
│   ├── modules/               # Core model components
│   │   ├── __init__.py
│   │   ├── embedding_module.py
│   │   ├── memory_updater.py
│   │   ├── memory.py
│   │   └── message_function.py
│   └── utils/                 # Model-specific utilities
│       ├── __init__.py
│       ├── data_processing.py
│       └── utils.py
├── StrGNN/                     # Structural Graph Neural Network model
│   ├── StrGNN_main.py         # Main model wrapper class
│   ├── detection/             # Detection algorithms
│   │   ├── Main.py
│   │   ├── Main_ori.py
│   │   ├── Main_statistic.py
│   │   ├── node2vec.py
│   │   ├── util_functions.py
│   │   └── data/              # Pre-computed data files
│   └── pytorch_DGCNN/         # DGCNN implementation
│       ├── main.py
│       ├── DGCNN_embedding.py
│       ├── mlp_dropout.py
│       ├── util.py
│       └── lib/               # C++ extensions
└── TADDY/                      # Temporal Anomaly Detection model
    ├── __init__.py
    ├── TADDY_main.py          # Main model wrapper class
    ├── codes/                 # Core implementation
    │   ├── __init__.py
    │   ├── AnomalyGeneration.py
    │   ├── BaseModel.py
    │   ├── Component.py
    │   ├── DynADModel.py
    │   ├── DynamicDatasetLoader.py
    │   ├── Settings.py
    │   ├── utils.py
    │   └── base_class/        # Base classes
    │       ├── dataset.py
    │       ├── evaluate.py
    │       ├── method.py
    │       ├── result.py
    │       └── setting.py
```

## Model Architecture Patterns

### 1. Wrapper Class Pattern

All models follow a consistent wrapper class pattern with a standardized interface:

```python
class ModelNameModel:
    def __init__(self, device, hyperparams, epoch_evaluation_metric):
        # Initialize hyperparameters and configuration
        
    def setup(self, df):
        # Process data and initialize model components
        
    def train(self):
        # Training loop implementation
        
    def inference(self, split="test"):
        # Inference on specified data split
        # Returns: (predictions, labels, inference_time)
```

### 2. Data Processing Patterns

Models handle different data input formats:
- **SLADE**: Uses pandas DataFrame with temporal edge data
- **StrGNN**: Uses polars DataFrame with snapshot-based data
- **TADDY**: Uses polars DataFrame with temporal snapshots

### 3. Device Management

All models support device specification (CPU/GPU) and handle tensor placement consistently:

```python
self.device = device
# Tensors moved to device during setup
tensor = tensor.to(self.device)
```

## Existing Models Overview

### SLADE (Temporal Graph Neural Network)

**Purpose**: Temporal anomaly detection using memory-augmented graph neural networks

**Key Features**:
- Memory-based temporal embeddings
- Contrastive learning approach
- Neighbor sampling for scalability
- Drift and recovery loss components

**Data Requirements**:
- Temporal edge data with timestamps
- Source/target node pairs
- Binary anomaly labels
- Train/test/validation masks

**Hyperparameters**:
```python
{
    "batch_size": 100,
    "num_neighbors": 20,
    "num_epoch": 10,
    "learning_rate": 3e-6,
    "memory_dim": 256,
    "message_dim": 128,
    "num_heads": 2,
    "drop_out": 0.1
}
```

### StrGNN (Structural Graph Neural Network)

**Purpose**: Structural anomaly detection using subgraph patterns

**Key Features**:
- Subgraph extraction and classification
- Node2Vec embeddings integration
- DGCNN-based graph classification
- Temporal window-based analysis

**Data Requirements**:
- Snapshot-based graph data
- Node embeddings (optional)
- Binary edge labels
- Temporal snapshots

**Hyperparameters**:
```python
{
    "batch_size": 32,
    "num_epoch": 20,
    "learning_rate": 1e-4,
    "hop": 1,
    "window_size": 5,
    "latent_dim": [32],
    "sortpooling_k": 30
}
```

### TADDY (Temporal Anomaly Detection with Dynamic Graphs)

**Purpose**: Transformer-based temporal anomaly detection

**Key Features**:
- Graph Transformer architecture
- Multi-type embeddings (structural, temporal, interaction)
- Negative sampling for training
- Eigen-decomposition for graph analysis

**Data Requirements**:
- Temporal graph snapshots
- Edge features (optional)
- Binary anomaly labels
- Train/test/validation splits

**Hyperparameters**:
```python
{
    "batch_size": 100,
    "num_neighbors": 5,
    "num_epoch": 20,
    "learning_rate": 0.001,
    "memory_dim": 256,
    "message_dim": 128,
    "num_heads": 2,
    "c": 0.15  # Model-specific parameter
}
```

## Model Interface Requirements

### Required Methods

Every model must implement these four core methods:

#### 1. `__init__(self, device, hyperparams, epoch_evaluation_metric)`

**Purpose**: Initialize model configuration and hyperparameters

**Parameters**:
- `device` (torch.device): Computation device (CPU/GPU)
- `hyperparams` (dict): Model-specific hyperparameters
- `epoch_evaluation_metric` (callable): Evaluation function for training

**Implementation Pattern**:
```python
def __init__(self, device, hyperparams, epoch_evaluation_metric):
    self.device = device
    self.epoch_evaluation_metric = epoch_evaluation_metric
    
    # Extract hyperparameters with defaults
    self.batch_size = hyperparams.get("batch_size", 100)
    self.num_epoch = hyperparams.get("num_epoch", 10)
    # ... other hyperparameters
    
    # Initialize placeholders for setup
    self.model = None
    self.optimizer = None
    # ... other components
```

#### 2. `setup(self, df)`

**Purpose**: Process input data and initialize model components

**Parameters**:
- `df` (pd.DataFrame or pl.DataFrame): Input graph data with required columns

**Required DataFrame Columns**:
- `src`: Source node IDs
- `tgt`: Target node IDs  
- `timestamp`: Edge timestamps
- `label`: Binary anomaly labels (0=normal, 1=anomaly)
- `train_mask`: Boolean mask for training edges
- `test_mask`: Boolean mask for test edges
- `val_mask`: Boolean mask for validation edges (optional)

**Implementation Pattern**:
```python
def setup(self, df):
    # Data validation
    required_cols = ["src", "tgt", "timestamp", "label", "train_mask", "test_mask"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Process data splits
    train_data = df[df["train_mask"]]
    test_data = df[df["test_mask"]]
    
    # Initialize model components
    self.model = ModelImplementation(...)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    # Setup validation
    self._ensure_setup_complete()
```

#### 3. `train(self)`

**Purpose**: Execute model training loop

**Implementation Pattern**:
```python
def train(self):
    self._ensure_setup()
    
    for epoch in range(self.num_epoch):
        self.model.train()
        epoch_loss = 0
        
        # Training loop
        for batch in self.data_loader:
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        
        # Validation evaluation
        if self.has_validation:
            val_preds, val_labels, _ = self.inference("val")
            val_score = self.epoch_evaluation_metric(val_labels, val_preds)
            logger.info(f"Epoch {epoch}: loss={epoch_loss:.4f}, val_score={val_score:.4f}")
```

#### 4. `inference(self, split="test")`

**Purpose**: Run inference on specified data split

**Parameters**:
- `split` (str): Data split to evaluate ("train", "val", "test")

**Returns**:
- `tuple[np.ndarray, np.ndarray, float]`: (predictions, labels, inference_time)

**Implementation Pattern**:
```python
def inference(self, split="test"):
    self._ensure_setup()
    start_time = time.time()
    
    self.model.eval()
    
    # Select data split
    if split == "train":
        data = self.train_data
    elif split == "val":
        data = self.val_data
    elif split == "test":
        data = self.test_data
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Run inference
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in data:
            pred = self.model(batch)
            predictions.append(pred.cpu().numpy())
            labels.append(batch.labels.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    inference_time = time.time() - start_time
    
    return predictions, labels, inference_time
```

### Helper Methods

#### `_ensure_setup(self)`

Validate that setup() has been called before training/inference:

```python
def _ensure_setup(self):
    if any(attr is None for attr in [self.model, self.optimizer, self.data]):
        raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")
```

## Pipeline Integration

Models are integrated into the DGADB pipeline through standardized pipeline scripts:

### Pipeline Script Pattern

```python
# src/dgadb/pipeline/pipeline_ModelName.py
import logging
from sklearn.metrics import roc_auc_score
from src.dgadb.data.dataset import load_df
from src.dgadb.models.ModelName.ModelName_main import ModelNameModel
from src.dgadb.preprocessing.splitting import generate_data_splits
from src.dgadb.utils import load_config

# Configuration
dataset = "dataset_name"
config = load_config(dataset)

# Data loading and preprocessing
data = load_df(dataset)
data = generate_data_splits(data, train_ratio=config["train_ratio"])

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperparams = config.get("hyperparams", {})
model = ModelNameModel(device, hyperparams, roc_auc_score)

# Training and evaluation
model.setup(data["edges"])
model.train()
preds, labels, inf_time = model.inference("test")

# Results
auc_score = roc_auc_score(labels, preds)
logger.info(f"Test ROC-AUC Score: {auc_score:.4f}")
```

### Integration Requirements

1. **Import Path**: Models must be importable as `from src.dgadb.models.ModelName.ModelName_main import ModelNameModel`
2. **Configuration**: Support loading hyperparameters from config files
3. **Evaluation**: Use sklearn.metrics.roc_auc_score as the standard evaluation metric
4. **Logging**: Use Python logging for progress and results reporting

## Adding New Models

### Step-by-Step Guide

#### 1. Create Model Directory Structure

```bash
mkdir -p src/dgadb/models/YourModel
touch src/dgadb/models/YourModel/__init__.py
touch src/dgadb/models/YourModel/YourModel_main.py
```

#### 2. Implement Model Wrapper Class

Create `YourModel_main.py` with the required interface:

```python
import logging
import time
import torch
import numpy as np
from typing import Any, Callable

logger = logging.getLogger(__name__)

class YourModelModel:
    """Your model description.
    
    Attributes:
        device (torch.device): Device to run computations on.
        epoch_evaluation_metric (callable): Function to evaluate model performance.
        # ... other attributes
    
    Args:
        device (torch.device): Device to run computations on.
        hyperparams (dict): Dictionary containing model hyperparameters.
        epoch_evaluation_metric (callable): Function that takes (labels, predictions)
            and returns a scalar evaluation metric.
    """
    
    def __init__(
        self,
        device: torch.device,
        hyperparams: dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        
        # Extract hyperparameters
        self.batch_size = hyperparams.get("batch_size", 100)
        self.num_epoch = hyperparams.get("num_epoch", 10)
        self.learning_rate = hyperparams.get("learning_rate", 0.001)
        # ... other hyperparameters
        
        # Initialize placeholders
        self.model = None
        self.optimizer = None
        self.data = None
    
    def setup(self, df) -> None:
        """Set up data processing and initialize the model."""
        # Validate input data
        required_cols = ["src", "tgt", "timestamp", "label", "train_mask", "test_mask"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Process data
        # ... your data processing logic
        
        # Initialize model
        # ... your model initialization
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
    
    def train(self) -> None:
        """Train the model."""
        self._ensure_setup()
        
        for epoch in range(self.num_epoch):
            # ... your training logic
            pass
    
    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray, float]:
        """Run inference on the specified data split."""
        self._ensure_setup()
        start_time = time.time()
        
        # ... your inference logic
        
        inference_time = time.time() - start_time
        return predictions, labels, inference_time
    
    def _ensure_setup(self) -> None:
        """Ensure setup() has been called before using the model."""
        if any(attr is None for attr in [self.model, self.optimizer, self.data]):
            raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")
```

#### 3. Create Pipeline Script

Create `src/dgadb/pipeline/pipeline_YourModel.py`:

```python
import logging
import torch
from sklearn.metrics import roc_auc_score
from src.dgadb.data.dataset import load_df
from src.dgadb.models.YourModel.YourModel_main import YourModelModel
from src.dgadb.preprocessing.splitting import generate_data_splits
from src.dgadb.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
dataset = "your_dataset"
config = load_config(dataset)

# Data loading
data = load_df(dataset)
data = generate_data_splits(
    data, 
    train_ratio=config["train_ratio"],
    val_ratio=config.get("val_ratio", None)
)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperparams = config.get("hyperparams", {})

model = YourModelModel(device, hyperparams, roc_auc_score)
model.setup(data["edges"])

# Training and evaluation
model.train()
preds, labels, inf_time = model.inference("test")

auc_score = roc_auc_score(labels, preds)
logger.info(f"Test ROC-AUC Score: {auc_score:.4f}")
```

#### 4. Add Configuration Support

Ensure your model supports configuration files in `configs/` directory with hyperparameters.

#### 5. Testing

Test your model with:
```bash
python -m src.dgadb.pipeline.pipeline_YourModel
```

## Best Practices

### 1. Code Organization

- **Modular Design**: Separate core model logic from the wrapper class
- **Clear Separation**: Keep data processing, model implementation, and evaluation separate
- **Reusable Components**: Create utility functions for common operations

### 2. Error Handling

```python
def setup(self, df):
    # Validate inputs
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame cannot be None or empty")
    
    # Check required columns
    required_cols = ["src", "tgt", "timestamp", "label", "train_mask", "test_mask"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        raise ValueError("Timestamp column must be numeric")
```

### 3. Memory Management

```python
def train(self):
    for epoch in range(self.num_epoch):
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(batch)
        loss = self.compute_loss(output, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Clear intermediate tensors
        del output, loss
        torch.cuda.empty_cache()  # If using GPU
```

### 4. Logging and Monitoring

```python
import logging
logger = logging.getLogger(__name__)

def train(self):
    logger.info(f"Starting training for {self.num_epoch} epochs...")
    
    for epoch in range(self.num_epoch):
        epoch_start = time.time()
        
        # Training logic
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{self.num_epoch}: "
                   f"loss={loss:.4f}, time={epoch_time:.2f}s")
```

### 5. Reproducibility

```python
def __init__(self, device, hyperparams, epoch_evaluation_metric):
    # Set random seeds for reproducibility
    self.seed = hyperparams.get("seed", 42)
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(self.seed)
```

## Common Patterns

### 1. Hyperparameter Management

```python
def __init__(self, device, hyperparams, epoch_evaluation_metric):
    # Use .get() with sensible defaults
    self.batch_size = hyperparams.get("batch_size", 100)
    self.learning_rate = hyperparams.get("learning_rate", 0.001)
    self.num_epoch = hyperparams.get("num_epoch", 10)
    
    # Log configuration
    logger.info(f"Initializing {self.__class__.__name__} with hyperparams: {hyperparams}")
```

### 2. Data Split Handling

```python
def setup(self, df):
    # Check for validation split
    self.has_val = "val_mask" in df.columns
    
    # Create data splits
    train_mask = df["train_mask"].to_numpy()
    test_mask = df["test_mask"].to_numpy()
    
    if self.has_val:
        val_mask = df["val_mask"].to_numpy()
        self.val_data = df[val_mask]
    
    self.train_data = df[train_mask]
    self.test_data = df[test_mask]
```

### 3. Device Management

```python
def setup(self, df):
    # Move model to device
    self.model = self.model.to(self.device)
    
    # Convert data to tensors on device
    self.train_tensors = {
        'src': torch.tensor(train_data['src'].values).to(self.device),
        'tgt': torch.tensor(train_data['tgt'].values).to(self.device),
        'timestamp': torch.tensor(train_data['timestamp'].values).to(self.device)
    }
```

### 4. Evaluation Integration

```python
def train(self):
    for epoch in range(self.num_epoch):
        # Training logic
        
        # Validation evaluation
        if self.has_val:
            val_preds, val_labels, _ = self.inference("val")
            val_score = self.epoch_evaluation_metric(val_labels, val_preds)
            logger.info(f"Epoch {epoch}: val_score={val_score:.4f}")
```

This documentation provides a comprehensive guide for understanding and extending the DGADB models directory. Follow these patterns and guidelines to ensure consistency and maintainability when adding new anomaly detection models to the framework.