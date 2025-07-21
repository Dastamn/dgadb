import polars as pl
from typing import Optional, Dict
import logging
logger = logging.getLogger(__name__)


def assign_snapshots(
    dfs: Dict[str, pl.DataFrame],
    window_size: int,
) -> Dict[str, pl.DataFrame]:

    def to_int_timestamp(col):
        if col.dtype in (pl.Date, pl.Datetime):
            return col.cast(pl.Int64)
        else:
            # already int or float
            return col

    min_time = to_int_timestamp(dfs["edges"]["timestamp"]).min()
    logger.debug(f"Initial min_time from edges: {min_time}")

    if "nodes" in dfs and "timestamp" in dfs["nodes"].columns:
        min_time = min(min_time, to_int_timestamp(
            dfs["nodes"]["timestamp"]).min())
        logger.debug(f"min_time after nodes: {min_time}")

    def assign(df):
        int_ts = to_int_timestamp(df["timestamp"])
        return df.with_columns([
            (((int_ts - min_time) // window_size)
                .cast(pl.Int64)).alias("snapshot_id")
        ])
    out_dfs = {"edges": assign(dfs["edges"])}
    if "nodes" in dfs and "timestamp" in dfs["nodes"].columns:
        out_dfs["nodes"] = assign(dfs["nodes"])
        logger.info("Assigned snapshot IDs to nodes and edges.")
    else:
        logger.info(
            "Assigned snapshot IDs to edges only (no node timestamps found).")
    return out_dfs
