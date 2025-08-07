import polars as pl
from typing import Optional, Literal
import logging
logger = logging.getLogger(__name__)

def normalize_timestamps(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    normalized_timestamp_col: str = "timestamp_norm",
    replace: bool = True
) -> pl.DataFrame:
    """
    Normalizes a timestamp column.

    1. Handles Date/Datetime types by converting to a high-resolution integer.
    2. For numeric types:
        - If integer or float with only whole numbers -> Preserves the integer sequence directly.
        - If "true" float (with fractional parts) -> Applies a scaling factor
          to convert all numbers to integers, preserving relative precision.

    Args:
        df: The input Polars DataFrame.
        timestamp_col: The name of the column containing the timestamps.
        normalized_timestamp_col: The name of the column containing the normalized timestamps.
        replace: Whether to replace the intial `timestamp_col` values or append a new `normalized_timestamp_col`.

    Returns:
        A new DataFrame with an added `normalized_timestamp_col` (UInt64) column.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame.")

    tar_col = timestamp_col if replace else normalized_timestamp_col
    dtype = df[timestamp_col].dtype
    timestamp_series = df[timestamp_col]

    # Date / Datetime
    if dtype in (pl.Date, pl.Datetime):
        normalized_timestamp_series = pl.col(timestamp_col).cast(pl.UInt64)

    # Numeric types (integer or float)
    elif dtype.is_numeric():
        # Check if the float column is effectively all integers
        is_integer_like = False
        if dtype in (pl.Float64, pl.Float32):
            # Check if all values are whole numbers
            if (timestamp_series == timestamp_series.floor()).all():
                is_integer_like = True

        if dtype.is_integer() or is_integer_like:
            # All integer-like numerics are treated as simple logical time sequences.
            normalized_timestamp_series = pl.col(timestamp_col).cast(pl.UInt64)
        else:
            # Floats with fractional parts -> applying relative scaling
            # Convert float to string to reliably find decimal places
            str_timestamp_series = timestamp_series.cast(pl.Utf8)

            # Find the max number of digits after the decimal point
            max_decimals = str_timestamp_series.str.split('.') \
                .list.last() \
                .str.len_chars() \
                .max()

            if max_decimals is None:
                max_decimals = 0

            scale_factor = 10 ** max_decimals
            normalized_timestamp_series = (
                pl.col(timestamp_col) * scale_factor).cast(pl.UInt64)

    else:
        raise TypeError(f"Unsupported timestamp dtype: {dtype}.")

    return df.with_columns(normalized_timestamp_series.alias(tar_col))


def generate_data_splits(
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


