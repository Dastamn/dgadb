from ..storage.graph import Graph
from typing import Dict, Optional
import polars as pl

def build_graph(data: Dict[str, pl.DataFrame], snapshot_split_map: Optional[pl.DataFrame] = None) -> Graph:
    df_edges = data["edges"]
    edges = {}
    nodes = {}
    timestamps = {}

    # there will always be edges, edge_ids, and edge timestamps
    edges["e_pairs"] = df_edges.select(["src", "tgt"]).to_torch(dtype=pl.Int64).T
    edges["e_id"] = df_edges.select("edge_id").to_torch(dtype=pl.Int64).flatten()
    timestamps["edges"] = df_edges.select(["edge_id", "timestamp"])
    edges["e_snapshot_id"] = df_edges.select("snapshot_id").to_torch(dtype=pl.Int64).flatten()


    # get n_id from edges, assuming no isolated nodes
    nodes["n_id"] = pl.concat([
        df_edges.get_column("src"),
        df_edges.get_column("tgt")
    ]).unique().sort().to_torch().flatten()

    feat_cols = [c for c in df_edges.columns if c.startswith("f")] #weak constraint
    if feat_cols:
         edges["e_feat"] = df_edges.select(feat_cols).to_torch(dtype=pl.Float32)

    if "label" in df_edges.columns:
        edges["e_label"] = df_edges.select("label").to_torch(dtype=pl.Int64).flatten()

    if "nodes" in data:
        df_nodes = data["nodes"]

        n_feat_cols = [c for c in df_nodes.columns if c.startswith("f")]
        if n_feat_cols:
            nodes["n_feat"] = df_nodes.select(n_feat_cols).to_torch(dtype=pl.Float32)

        if "node_type" in df_nodes.columns:
            nodes["n_type"] = df_nodes.select("node_type").to_torch(dtype=pl.Int64).flatten()
        
        if "timestamp" in df_nodes.columns:
            timestamps["nodes"] = df_nodes.select(["node_id", "timestamp"])
        
        if "snapshot_id" in df_nodes.columns:
            nodes["n_snapshot_id"] = df_nodes.select("snapshot_id").to_torch(dtype=pl.Int64).flatten()


    return Graph(nodes=nodes, edges=edges, timestamps=timestamps, snapshot_split_map=snapshot_split_map)