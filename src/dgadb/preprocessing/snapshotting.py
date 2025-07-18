import polars as pl
from typing import Optional, Dict

def assign_snapshots(
    dfs: Dict[str, pl.DataFrame],
    window_size: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    def to_int_timestamp(col):
        if col.dtype in (pl.Date, pl.Datetime):
            return col.cast(pl.Int64)
        else:
            # already int or float
            return col
        
    min_time = to_int_timestamp(dfs["edges"]["timestamp"]).min()

    if "nodes" in dfs and "timestamp" in dfs["nodes"].columns:
        min_time = min(min_time, to_int_timestamp(dfs["nodes"]["timestamp"]).min())

    def assign(df):
        int_ts = to_int_timestamp(df["timestamp"])
        return df.with_columns([
            (((int_ts - min_time) // window_size)
                .cast(pl.Int64)).alias("snapshot_id")
        ])
    out_dfs = {"edges" : assign(dfs["edges"])}
    if "nodes" in dfs and "timestamp" in dfs["nodes"].columns:
        out_dfs["nodes"] = assign(dfs["nodes"])

    return out_dfs
