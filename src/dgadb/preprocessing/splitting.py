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
