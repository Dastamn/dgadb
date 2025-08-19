import torch
import numpy as np
from ..storage.graph import Graph
from typing import Dict, Optional
import polars as pl
import logging

logger = logging.getLogger(__name__)


# TODO @Dastamn: Refactor
def build_graph(
    data: Dict[str, pl.DataFrame], node_features: np.ndarray | None, edge_features: np.ndarray | None, window_size: int
) -> Graph:
    logger.info("Building Graph object from dataframes.")
    logger.debug(f"Data keys: {list(data.keys())}")
    logger.debug(f"Edges shape: {data['edges'].shape}")
    if "nodes" in data:
        logger.debug(f"Nodes shape: {data['nodes'].shape}")

    df_edges = data["edges"]
    edges = {}
    nodes = {}
    # timestamps = {}

    # there will always be edges, edge_ids, and edge timestamps
    edges["e_pairs"] = df_edges.select(["src", "tgt"]).to_torch(dtype=pl.Int64).T
    edges["e_id"] = df_edges.select("edge_id").to_torch(dtype=pl.Int64).flatten()
    edges["e_timestamp"] = df_edges.select("timestamp").to_torch(dtype=pl.Int64).flatten()
    edges["e_snapshot_id"] = df_edges.select("snapshot_id").to_torch(dtype=pl.Int64).flatten()
    edges["e_train_mask"] = df_edges.select("train_mask").to_torch(dtype=pl.Boolean).flatten()
    edges["e_test_mask"] = df_edges.select("test_mask").to_torch(dtype=pl.Boolean).flatten()
    logger.info(f"Edges, their timestamps, and their snapshot_ids are loaded.")

    # get n_id from edges, assuming no isolated nodes
    nodes["n_id"] = (
        pl.concat([df_edges.get_column("src"), df_edges.get_column("tgt")]).unique().sort().to_torch().flatten()
    )

    if edge_features is not None:
        edges["e_feat"] = torch.from_numpy(edge_features)
    if "val_mask" in df_edges.columns:
        logger.info("Validation mask column found.")
        edges["e_val_mask"] = df_edges.select("val_mask").to_torch(dtype=pl.Boolean).flatten()
    if "label" in df_edges.columns:
        logger.info("Edge label column found.")
        edges["e_label"] = df_edges.select("label").to_torch(dtype=pl.Int64).flatten()
    else:
        logger.info("No edge label column found.")

    if "nodes" in data:
        df_nodes = data["nodes"]

        if node_features is not None:
            nodes["n_feat"] = torch.from_numpy(node_features)

        if "node_type" in df_nodes.columns:
            logger.info(f"Found node type column.")
            nodes["n_type"] = df_nodes.select("node_type").to_torch(dtype=pl.Int64).flatten()
        else:
            logger.info(f"No node type column found.")

        if "timestamp" in df_nodes.columns:
            logger.info(f"Found node timestamp column.")
            # timestamps["nodes"] = df_nodes.select(["node_id", "timestamp"])
        else:
            logger.info(f"No node timestamp column found.")

        if "snapshot_id" in df_nodes.columns:
            nodes["n_snapshot_id"] = df_nodes.select("snapshot_id").to_torch(dtype=pl.Int64).flatten()

    return Graph(nodes=nodes, edges=edges, window_size=window_size)


def build_graph_from_temporal(temporal_graph) -> Graph:
    """
    Convert TemporalGraphData to Graph object.
    
    This function provides the proper abstraction layer for converting
    preprocessed temporal graph data into the final Graph object that
    models expect, following the "black box" principle.
    
    Args:
        temporal_graph: TemporalGraphData object from the preprocessing pipeline
        
    Returns:
        Graph: Properly formatted Graph object ready for model consumption
    """
    logger.info("Converting TemporalGraphData to Graph object...")
    
    # Prepare edge data
    edges = {
        "e_pairs": torch.stack([temporal_graph.src, temporal_graph.tgt]),
        "e_label": temporal_graph.edge_labels,
        "e_train_mask": temporal_graph.train_mask,
        "e_val_mask": temporal_graph.val_mask,
        "e_test_mask": temporal_graph.test_mask,
        "e_timestamp": temporal_graph.t,
    }
    
    # Add edge features if available
    if temporal_graph.msg.numel() > 0:
        edges["e_feat"] = temporal_graph.msg
        logger.info(f"Added edge features with shape {temporal_graph.msg.shape}")
    
    # Prepare node data
    nodes = {}
    if temporal_graph.node_attr is not None:
        nodes["n_feat"] = temporal_graph.node_attr
        logger.info(f"Added node features with shape {temporal_graph.node_attr.shape}")
    else:
        # Create identity matrix for node features if none provided
        num_nodes = temporal_graph.num_nodes
        nodes["n_feat"] = torch.eye(num_nodes)
        logger.info(f"Created identity node features for {num_nodes} nodes")
    
    # Add node labels if available
    if temporal_graph.node_labels is not None:
        nodes["n_label"] = temporal_graph.node_labels
    
    # Create and return Graph object
    graph = Graph(nodes=nodes, edges=edges)
    
    logger.info(f"Successfully created Graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
    logger.info(f"Train edges: {edges['e_train_mask'].sum().item()}, "
                f"Val edges: {edges['e_val_mask'].sum().item()}, "
                f"Test edges: {edges['e_test_mask'].sum().item()}")
    
    return graph
