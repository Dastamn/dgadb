"""Load experiment results from benchmark runs.

This module provides functions for scanning and loading evaluation.json files
from the experiment-results directory structure.
"""

import json
import os
import re
from pathlib import Path

import polars as pl


def parse_variant_name(variant: str) -> dict:
    """Parse a variant name into its components.

    Args:
        variant: Variant string like "random_tr0.0_v0.1_te0.1_dur0.5" or "clean".

    Returns:
        Dictionary with parsed components.
    """
    if variant == "clean":
        return {
            "anom_type": None,
            "train_ratio": 0.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "duration": None,
        }

    # Try new format: {anom_type}_tr{tr}_v{v}_te{te}_dur{dur}
    match = re.match(
        r"^(\w+)_tr([\d.]+)_v([\d.]+)_te([\d.]+)_dur([\d.]+)$",
        variant,
    )
    if match:
        return {
            "anom_type": match.group(1),
            "train_ratio": float(match.group(2)),
            "val_ratio": float(match.group(3)),
            "test_ratio": float(match.group(4)),
            "duration": float(match.group(5)),
        }

    # Try old format: {anom_type}_tr{tr}_v{v}_te{te}_{size}
    match = re.match(
        r"^(\w+)_tr([\d.]+)_v([\d.]+)_te([\d.]+)_(small|medium|large)$",
        variant,
    )
    if match:
        duration_map = {"small": 0.1, "medium": 0.5, "large": 1.0}
        return {
            "anom_type": match.group(1),
            "train_ratio": float(match.group(2)),
            "val_ratio": float(match.group(3)),
            "test_ratio": float(match.group(4)),
            "duration": duration_map.get(match.group(5)),
        }

    # Fallback: just extract what we can
    return {
        "anom_type": variant,
        "train_ratio": None,
        "val_ratio": None,
        "test_ratio": None,
        "duration": None,
    }


def load_experiment_results(
    experiment_dir: str = "experiment-results",
    methods: list[str] | None = None,
    datasets: list[str] | None = None,
    experiments: list[str] | None = None,
) -> pl.DataFrame:
    """Load experiment results from evaluation.json files.

    Scans the experiment-results directory structure and parses all
    evaluation.json files into a polars DataFrame.

    Directory structure expected:
        {experiment_dir}/{experiment}/{method}/{variant}/evaluation.json

    Args:
        experiment_dir: Base directory containing experiment results.
        methods: Optional list of methods to include (filters results).
        datasets: Optional list of datasets to include (filters results).
        experiments: Optional list of experiment names to include.

    Returns:
        DataFrame with columns: experiment, method, variant, anom_type,
        train_ratio, val_ratio, test_ratio, duration, roc_auc, average_precision,
        max_f1, accuracy, balanced_accuracy.
    """
    if not os.path.exists(experiment_dir):
        return pl.DataFrame()

    records = []
    base_path = Path(experiment_dir)

    for eval_file in base_path.rglob("evaluation.json"):
        parts = eval_file.relative_to(base_path).parts

        if len(parts) < 3:
            continue

        # Parse path: experiment/method/variant/evaluation.json
        experiment = parts[0]
        method = parts[1]
        variant = parts[2]

        # Apply filters
        if experiments is not None and experiment not in experiments:
            continue
        if methods is not None and method not in methods:
            continue

        # Parse variant components
        variant_info = parse_variant_name(variant)

        # Infer dataset from experiment name if possible
        # Common pattern: dataset-name or dataset-name-suffix
        dataset = experiment  # Default to experiment name

        if datasets is not None and dataset not in datasets:
            # Try to match dataset as prefix
            matched = False
            for ds in datasets:
                if experiment.startswith(ds):
                    dataset = ds
                    matched = True
                    break
            if not matched:
                continue

        # Load JSON
        try:
            with open(eval_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Extract metrics from summary
        summary = data.get("summary", {})

        record = {
            "experiment": experiment,
            "dataset": dataset,
            "method": method,
            "variant": variant,
            "anom_type": variant_info["anom_type"],
            "train_ratio": variant_info["train_ratio"],
            "val_ratio": variant_info["val_ratio"],
            "test_ratio": variant_info["test_ratio"],
            "duration": variant_info["duration"],
            "roc_auc": summary.get("roc_auc", {}).get("mean"),
            "average_precision": summary.get("average_precision", {}).get("mean"),
            "max_f1": summary.get("max_f1", {}).get("mean"),
            "accuracy": summary.get("accuracy", {}).get("mean"),
            "balanced_accuracy": summary.get("balanced_accuracy", {}).get("mean"),
        }

        records.append(record)

    if not records:
        return pl.DataFrame()

    return pl.DataFrame(records)


def load_single_result(eval_path: str) -> dict | None:
    """Load a single evaluation.json file.

    Args:
        eval_path: Path to evaluation.json file.

    Returns:
        Dictionary with results, or None if loading fails.
    """
    try:
        with open(eval_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_available_experiments(experiment_dir: str = "experiment-results") -> list[str]:
    """Get list of available experiment names.

    Args:
        experiment_dir: Base directory containing experiment results.

    Returns:
        List of experiment directory names.
    """
    if not os.path.exists(experiment_dir):
        return []

    return [
        d
        for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
    ]


def get_available_methods(experiment_dir: str = "experiment-results") -> list[str]:
    """Get list of available methods across all experiments.

    Args:
        experiment_dir: Base directory containing experiment results.

    Returns:
        List of unique method names.
    """
    if not os.path.exists(experiment_dir):
        return []

    methods = set()
    for exp_dir in Path(experiment_dir).iterdir():
        if exp_dir.is_dir():
            for method_dir in exp_dir.iterdir():
                if method_dir.is_dir():
                    methods.add(method_dir.name)

    return sorted(methods)
