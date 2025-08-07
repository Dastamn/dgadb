import polars as pl
from typing import Optional, Dict
import logging
logger = logging.getLogger(__name__)


def assign_snapshots(
    dfs: dict[str, pl.DataFrame],
    snapshot_size: int,
    temporal_snapshots: bool = False,
) -> dict[str, pl.DataFrame]:
    """
    Assigns snapshot IDs to edges (and optionally nodes) in a dynamic graph DataFrame.

    Two snapshotting strategies are supported:

    1. Temporal snapshots (`temporal_snapshots=True`):
        - Edges are grouped into snapshots based on their timestamp.
        - Each snapshot contains all edges whose timestamps fall within a fixed window (`snapshot_size`).
        - Ensures no overlap in snapshot IDs between train, val, and test splits.

    2. Structural snapshots (`temporal_snapshots=False`):
        - Edges are grouped into snapshots by fixed edge count (`snapshot_size`).
        - The first `snapshot_size` edges form snapshot 0, the next `snapshot_size` form snapshot 1, etc
        - Ensures no overlap in snapshot IDs between train, val, and test splits.

    Args:
        dfs: Dictionary containing at least an "edges" Polars DataFrame, and optionally a "nodes" DataFrame.
             The "edges" DataFrame must have boolean columns "train_mask" and "test_mask", and optionally "val_mask".
        snapshot_size: The window size (for temporal) or edge count (for structural) for each snapshot.
        temporal_snapshots: If True, use temporal snapshotting; if False, use structural snapshotting.

    Returns:
        The input dictionary with "edges" (and optionally "nodes") DataFrame(s) updated to include a "snapshot_id" column.
        Snapshot IDs are unique and non-overlapping across splits.
    """
    
    def _temporal_snapshots(dfs:dict[str, pl.DataFrame], snapshot_size: int):

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
                (((int_ts - min_time) // snapshot_size)
                    .cast(pl.Int64)).alias("snapshot_id")
            ])
        
        
        train = dfs["edges"].filter(pl.col("train_mask"))
        val = dfs["edges"].filter(pl.col("val_mask")) if "val_mask" in dfs["edges"].columns else pl.DataFrame()
        test = dfs["edges"].filter(pl.col("test_mask"))

        train = assign(train)
        df_ordered = [train]
        max_snap_train = train["snapshot_id"].max()
        max_snap_val = None
        if val.height > 0:
            min_snap_val = train["snapshot_id"].min()
            val = assign(val)
            if max_snap_train == min_snap_val:
                val = val.with_columns((pl.col("snapshot_id") + 1).alias("snapshot_id"))
            df_ordered.append(val)
            max_snap_val = val["snapshot_id"].max()
        test = assign(test)
        min_snap_test = test["snapshot_id"].min()
        if max_snap_val is None:
            if max_snap_train == min_snap_test:
                test = test.with_columns((pl.col("snapshot_id") + 1).alias("snapshot_id"))
        else:
            if max_snap_val == min_snap_test:
                test = test.with_columns((pl.col("snapshot_id") + 1).alias("snapshot_id"))
        df_ordered.append(test)
        

        dfs["edges"] = pl.concat(df_ordered, how="vertical")

        if "nodes" in dfs and "timestamp" in dfs["nodes"].columns:
            dfs["nodes"] = assign(dfs["nodes"])
            logger.info("Assigned snapshot IDs to nodes and edges.")
        else:
            logger.info(
                "Assigned snapshot IDs to edges only (no node timestamps found).")
        return dfs
    
    def _structural_snapshots(dfs: dict[str, pl.DataFrame], snapshot_size: int):
        def assign_structural(df: pl.DataFrame, start_snap_id):
            if df.height == 0:
                return df
            # Assign snapshot_id based on row index, incremented by start_snap_id
            return (
                df.with_row_index("row_idx")
                .with_columns(
                    ((pl.col("row_idx") // snapshot_size) + start_snap_id).alias("snapshot_id")
                )
                .drop("row_idx")
            )

        # Split by mask
        train = dfs["edges"].filter(pl.col("train_mask"))
        val = dfs["edges"].filter(pl.col("val_mask")) if "val_mask" in dfs["edges"].columns else pl.DataFrame()
        test = dfs["edges"].filter(pl.col("test_mask"))

        # Assign snapshot_ids, increment for each split
        snap_id = 0
        train = assign_structural(train, snap_id)
        snap_id += train["snapshot_id"].max() + 1 if train.height > 0 else 0

        if val.height > 0:
            val = assign_structural(val, snap_id)
            snap_id += val["snapshot_id"].max() + 1 if val.height > 0 else 0

        test = assign_structural(test, snap_id)

        # Concate in order train, val (if exists), test
        df_ordered = [train]
        if val.height > 0:
            df_ordered.append(val)
        df_ordered.append(test)
        dfs["edges"] = pl.concat(df_ordered, how="vertical")

        logger.info("Assigned structural snapshot_ids to edges")
        return dfs
        
    
    if temporal_snapshots:
        return _temporal_snapshots(dfs, snapshot_size)
    else:
        return _structural_snapshots(dfs, snapshot_size)
    #TODO @tobiasmoller27: Make structural work for nodes as well