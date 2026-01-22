# Running Experiments
All experiments can be executed using `runner.py`.
This script serves as a simple entry point for testing different methods on different datasets.

---

## How to Run

```bash
python runner.py <method> <dataset>
```

### Arguments

* **`method`**
  Must be written in lowercase and must match one of the method names defined in the `main` function inside `runner.py`.

* **`dataset`**
  Must match one of the dataset names listed in:


```
config/dataset
```

Example command to run from the root of the dgadb project (i.e., in the directory that contains the `src` folder):
```bash
BASE_PATH=$PWD python -m src.dgadb.experiment.runner --method sad --dataset bitcoin-alpha
```
---

## Configuration

At the moment, all parameters must be edited directly in the code.
More flexible configuration handling may be added in the future.

### Anomaly Injection

```python
anom_config = {
    "anom_type": "structural",
    "anom_test_ratio": 0.1,
    "anom_val_ratio": 0.1,
}
```

This is the most important configuration, as it controls how many anomalies are injected into the data.
It may change as new anomaly types are introduced.

---

### Snapshot Generation

```python
snapshot_config = {
    "strategy": "window",
    "window_size": 2000,
    "include_cumulative": True
}
```

---

## Hyperparameter Tuning

Hyperparameter tuning is not supported for methods executed through `runner.py`.

All hyperparameter tuning must be performed using:

```
tune.py
```

---

## Issues

Currently, STRGNN behaves inconsistently:

* It crashes when anomalies are injected.
* It works correctly on the previously built datasets used to generate the Excel results table.

The cause of this behavior is still under investigation.
