from typing import Optional
import polars as pl
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def generate_data_splits(
    dfs: Dict[str, pl.DataFrame],
    snapshot_col: str = "snapshot_id",
    train_ratio: float = 0.7,
    val_ratio: Optional[float] = None
) -> pl.DataFrame:

    logger.info(
        f"Generating data splits: train_ratio={train_ratio}, val_ratio={val_ratio}, snapshot_col={snapshot_col}")

    ids = dfs["edges"].select(snapshot_col).unique()
    if "nodes" in dfs and snapshot_col in dfs["nodes"].columns:
        ids = ids.vstack(dfs["nodes"].select(snapshot_col).unique())
    ids = ids.unique().sort(by=snapshot_col)

    n = ids.height
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio) if val_ratio else 0
    n_test = n - n_train - n_val

    splits = (
        ["train"] * n_train +
        (["val"] * n_val if n_val > 0 else []) +
        ["test"] * n_test
    )[:n]

    logger.info(
        f"Assigned splits: {n_train} train, {n_val} val, {n_test} test (total {n})")

    ids = ids.with_columns([
        pl.Series("split", splits)
    ])
    return ids

# TODO add choice to use fixed snapshot numbers instead of ratios


def generate_data_splits_2(
    edges: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: Optional[float] = None,
    timestamp_col: str = "timestamp"
) -> pl.DataFrame:
    """
    Splits a DataFrame of edges into train, validation, and test sets chronologically.

    Args:
        edges: DataFrame of timestamped edges.
        train_ratio: The proportion of data to allocate to the training set.
        val_ratio: The proportion of data to allocate to the validation set.
                   If None, no validation set is created.
        timestamp_col: The name of the timestamp column. Defaults to "timestamp".

    Returns:
        The original DataFrame with added boolean mask columns
        "train_mask", "test_mask", and "val_mask" (only if `val_ratio` is provided).
    """
    if timestamp_col not in edges.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame.")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("'train_ratio' must be between 0 and 1.")

    if val_ratio is not None and not 0.0 < val_ratio < 1.0:
        raise ValueError("'val_ratio' must be between 0 and 1 if provided.")

    if val_ratio is not None and (train_ratio + val_ratio >= 1.0):
        raise ValueError("(train_ratio + val_ratio) must be less than 1.")

    edges_sorted = edges.sort(timestamp_col)

    n_total = len(edges_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio) if val_ratio is not None else 0
    n_test = n_total - n_train - n_val

    train_mask = [True] * n_train + [False] * (n_val + n_test)
    test_mask = [False] * (n_train + n_val) + [True] * n_test

    edges_with_splits = edges_sorted.with_columns([
        pl.Series("train_mask", train_mask),
        pl.Series("test_mask", test_mask)
    ])

    if val_ratio is not None:
        val_mask = [False] * n_train + [True] * n_val + [False] * n_test
        edges_with_splits = edges_with_splits.with_columns(
            pl.Series("val_mask", val_mask))

    return edges_with_splits
