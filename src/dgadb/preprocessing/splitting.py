from typing import Optional
import polars as pl
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def generate_data_splits(
    dfs: Dict[str, pl.DataFrame],
    train_ratio: float = 0.7,
    val_ratio: Optional[float] = None
) -> pl.DataFrame:

    logger.info(
        f"Generating data splits: train_ratio={train_ratio}, val_ratio={val_ratio}")

    timestamps = dfs["edges"]["timestamp"]

    n = len(timestamps)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio) if val_ratio else 0
    n_test = n - n_train - n_val

    train_mask = [True] * n_train + [False] * (n - n_train)
    val_mask = [False] * n
    test_mask = [False] * n

    if n_val > 0:
        val_mask[n_train:n_train + n_val] = [True] * n_val
        test_mask[n_train + n_val:] = [True] * n_test
    else:
        test_mask[n_train:] = [True] * n_test

    # Add masks to df
    dfs["edges"] = dfs["edges"].with_columns([
        pl.Series("train_mask", train_mask),
        pl.Series("test_mask", test_mask)
    ])
    if n_val > 0:
        dfs["edges"] = dfs["edges"].with_columns(
            [pl.Series("val_mask", val_mask)])

    logger.info(
        f"Assigned splits: {n_train} train, {n_val} val, {n_test} test (total {n})")
    return dfs
# TODO add choice to use fixed snapshot numbers instead of ratios
