import polars as pl
import torch

def make_undirected_train(dfs: dict[str, pl.DataFrame]):
    edges = dfs["edges"]

    # Apply edge ordering only to training edges
    edges = edges.with_columns([
        # Create new src and tgt columns where smaller node ID is always src
        pl.when(pl.col("train_mask"))
        .then(pl.min_horizontal(["src", "tgt"]))
        .otherwise(pl.col("src"))
        .alias("new_src"),

        pl.when(pl.col("train_mask"))
        .then(pl.max_horizontal(["src", "tgt"]))
        .otherwise(pl.col("tgt"))
        .alias("new_tgt")
    ]).with_columns([
        # Replace original src/tgt with the ordered versions
        pl.col("new_src").alias("src"),
        pl.col("new_tgt").alias("tgt")
    ]).drop(["new_src", "new_tgt"])
    dfs["edges"] = edges

    return dfs

def make_undirected(dfs: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    edges = dfs["edges"]

    # Always order src and tgt so that src <= tgt
    edges = edges.with_columns([
        pl.min_horizontal(["src", "tgt"]).alias("src"),
        pl.max_horizontal(["src", "tgt"]).alias("tgt"),
    ])
    dfs["edges"] = edges
    return dfs

def remove_self_loops_train(dfs: dict[str, pl.DataFrame]):
    edges = dfs["edges"]

    # Remove self-loops only from training edges
    edges = edges.filter(
        (~pl.col("train_mask")) | 
        (pl.col("train_mask") & (pl.col("src") != pl.col("tgt")))
    )
    dfs["edges"] = edges

    return dfs

def remove_self_loops(dfs: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    edges = dfs["edges"]

    # Remove all self-loops (src == tgt)
    edges = edges.filter(pl.col("src") != pl.col("tgt"))
    dfs["edges"] = edges
    return dfs

def remove_duplicates_train(dfs: dict[str, pl.DataFrame]):
    edges = dfs["edges"]

    # Split into training and and non-training edges
    train_edges = edges.filter(pl.col("train_mask"))
    non_train_edges = edges.filter(~pl.col("train_mask"))

    # Remove duplicates from training edges based on src-tgt pairs
    train_edges_unique = train_edges.unique(subset=["src", "tgt"], keep="first")

    # Concat back together
    edges = pl.concat([train_edges_unique, non_train_edges])

    dfs["edges"] = edges.sort("timestamp")

    return dfs


def remove_duplicates(dfs: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    edges = dfs["edges"]

    # Remove duplicates based on src-tgt pairs, keeping the first occurrence (by timestamp if available)
    edges = edges.unique(subset=["src", "tgt"], keep="first")
    dfs["edges"] = edges.sort("timestamp")
    return dfs

def reindex_nodes(dfs: dict[str, pl.DataFrame]):
    """
    Reindex node IDs to ensure they go from 0 to n-1
    """
    edges = dfs["edges"]

    # Get all unique node IDs from both src and tgt columns
    unique_nodes = pl.concat([
        edges.select(pl.col("src").alias("node_id")),
        edges.select(pl.col("tgt").alias("node_id"))
    ]).unique().sort("node_id")

    # Create mapping from original_id to new_id (0 to n-1)
    node_mapping = unique_nodes.with_row_index("new_id")

    # Get the old and new values as lists
    old_values = node_mapping["node_id"].to_list()
    new_values = node_mapping["new_id"].to_list()

    # Reindex nodes
    edges_reindexed = edges.with_columns([
        pl.col("src").replace_strict(old_values, new_values, default=None).alias("src"),
        pl.col("tgt").replace_strict(old_values, new_values, default=None).alias("tgt")
    ])

    edges_reindexed = edges_reindexed.drop("edge_id").with_row_index("edge_id")

    dfs["edges"] = edges_reindexed.sort("timestamp")

    return dfs



