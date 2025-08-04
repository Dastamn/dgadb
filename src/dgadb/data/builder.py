import torch
import numpy as np
from ..storage.graph import Graph
from typing import Dict, Optional
import polars as pl
import logging
logger = logging.getLogger(__name__)


# TODO @Dastamn: Refactor
def build_graph(data: Dict[str, pl.DataFrame], node_features: np.ndarray | None, edge_features: np.ndarray | None, window_size: int) -> Graph:
    logger.info("Building Graph object from dataframes.")
    logger.debug(f"Data keys: {list(data.keys())}")
    logger.debug(f"Edges shape: {data['edges'].shape}")
    if 'nodes' in data:
        logger.debug(f"Nodes shape: {data['nodes'].shape}")

    df_edges = data["edges"]
    edges = {}
    nodes = {}
    # timestamps = {}

    # there will always be edges, edge_ids, and edge timestamps
    edges["e_pairs"] = df_edges.select(
        ["src", "tgt"]).to_torch(dtype=pl.Int64).T
    edges["e_id"] = df_edges.select(
        "edge_id").to_torch(dtype=pl.Int64).flatten()
    # timestamps["edges"] = df_edges.select(["edge_id", "timestamp"])
    edges["e_snapshot_id"] = df_edges.select(
        "snapshot_id").to_torch(dtype=pl.Int64).flatten()
    logger.info(f"Edges, their timestamps, and their snapshot_ids are loaded.")

    # get n_id from edges, assuming no isolated nodes
    nodes["n_id"] = pl.concat([
        df_edges.get_column("src"),
        df_edges.get_column("tgt")
    ]).unique().sort().to_torch().flatten()

    # weak constraint
    # feat_cols = [c for c in df_edges.columns if c.startswith("f")]
    # if feat_cols:
    #     logger.info(f"Found edge feature columns: {feat_cols}")
    #     edges["e_feat"] = df_edges.select(feat_cols).to_torch(dtype=pl.Float32)
    # else:
    #     logger.info(f"No edge feature columns found.")

    if edge_features is not None:
        edges["e_feat"] = torch.from_numpy(edge_features)

    if "label" in df_edges.columns:
        logger.info("Edge label column found.")
        edges["e_label"] = df_edges.select(
            "label").to_torch(dtype=pl.Int64).flatten()
    else:
        logger.info("No edge label column found.")

    if "nodes" in data:
        df_nodes = data["nodes"]

        # n_feat_cols = [c for c in df_nodes.columns if c.startswith("f")]
        # if n_feat_cols:
        #     logger.info(f"Found node feature columns: {n_feat_cols}")
        #     nodes["n_feat"] = df_nodes.select(
        #         n_feat_cols).to_torch(dtype=pl.Float32)
        # else:
        #     logger.info("No node feature columns found.")

        if node_features is not None:
            nodes["n_feat"] = torch.from_numpy(node_features)

        if "node_type" in df_nodes.columns:
            logger.info(f"Found node type column.")
            nodes["n_type"] = df_nodes.select(
                "node_type").to_torch(dtype=pl.Int64).flatten()
        else:
            logger.info(f"No node type column found.")

        if "timestamp" in df_nodes.columns:
            logger.info(f"Found node timestamp column.")
            # timestamps["nodes"] = df_nodes.select(["node_id", "timestamp"])
        else:
            logger.info(f"No node timestamp column found.")

        if "snapshot_id" in df_nodes.columns:
            nodes["n_snapshot_id"] = df_nodes.select(
                "snapshot_id").to_torch(dtype=pl.Int64).flatten()

    return Graph(nodes=nodes, edges=edges, window_size=window_size)
