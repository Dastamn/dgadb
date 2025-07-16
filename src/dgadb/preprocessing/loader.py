import yaml
import os
from typing import Dict
import polars as pl
from .normalization import get_normalizer
from collections import defaultdict
import torch
from src.dgadb.storage.graph import Graph

def load_config(name: str) -> dict:
    base_path = os.environ["BASE_PATH"]
    yaml_path = os.path.join(base_path, "data", name,"normalization.yaml")

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
    
def load_df(name: str) -> Dict[str, pl.DataFrame]:
    base_path = os.environ["BASE_PATH"]
    ds_dir_path = os.path.join(base_path, "data", name)

    edge_files = [
        "edges",
        "edge_labels",
        "edge_types",
        "edge_features_cat",
        "edge_features_num",
        "edge_features_str"
    ]
    
    node_files = [
        "node_labels",
        "node_types",
        "node_timestamps",
        "node_features_cat",
        "node_features_num",
        "node_features_str"
    ]

    req_file = os.path.join(ds_dir_path, "edges.parquet")
    if not os.path.exists(req_file):
        raise FileNotFoundError(f"Required file not found: {req_file}")

    df_edges = pl.read_parquet(req_file)

    for f_name in edge_files[1:]:
        f_path = os.path.join(ds_dir_path, f"{f_name}.parquet")
        if os.path.exists(f_path):
            df_extra = pl.read_parquet(f_path)
            if {"edge_id", "feature_id", "value"}.issubset(df_extra.columns):
                suffix = f_name.split("_")[-1]  # cat/num/str
                df_extra = df_extra.pivot(index="edge_id", on="feature_id", values="value")
                feature_cols = [col for col in df_extra.columns if col != "edge_id"]
                df_extra = df_extra.rename({
                    col: f"f{col}_{suffix}" for col in feature_cols
                })
            df_edges = df_edges.join(df_extra, on="edge_id", how="left")

    result = {"edges": df_edges}
    print(df_edges)

    node_dfs = []
    for f_name in node_files:
        f_path = os.path.join(ds_dir_path, f"{f_name}.parquet")
        if os.path.exists(f_path):
            df = pl.read_parquet(f_path)
            if {"node_id", "feature_id", "value"}.issubset(df.columns):
                suffix = f_name.split("_")[-1]
                df = df.pivot(index="node_id", on="feature_id", values="value")
                feature_cols = [col for col in df.columns if col != "node_id"]
                df = df.rename({
                    col: f"f{col}_{suffix}" for col in feature_cols
                })
            node_dfs.append(df)

    if node_dfs:
        df_nodes = node_dfs[0]
        for df in node_dfs[1:]:
            df_nodes = df_nodes.join(df, on="node_id", how="left")
        result["nodes"] = df_nodes

    return result

def apply_normalizers(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    for norm_type, columns in config.items():
        valid_columns = [col for col in columns if col in df.columns]
        missing_columns = set(columns) - set(valid_columns)
        for missing in missing_columns:
            print(f"Warning: column '{missing}' not found. Skipping.")
        if not valid_columns:
            continue
        normalizer = get_normalizer(norm_type, norm_type in {"bow", "tfidf"})
        normalizer.fit(df, valid_columns)
        df = normalizer.transform(df)
    return df

def normalize_dataframes(data: Dict[str, pl.DataFrame], config: dict) -> Dict[str, pl.DataFrame]:
    if "nodes" in data and config.get("nodes", {}).get("normalizers"):
        data["nodes"] = apply_normalizers(data["nodes"], config["nodes"]["normalizers"])

    if "edges" in data and config.get("edges", {}).get("normalizers"):
        data["edges"] = apply_normalizers(data["edges"], config["edges"]["normalizers"])

    return data

def build_graph(data: Dict[str, pl.DataFrame]) -> Graph:
    df_edges = data["edges"]
    edges = {}
    nodes = {}
    timestamps = {}

    # there will always be edges, edge_ids, and edge timestamps
    edges["e_pairs"] = df_edges.select(["src", "tgt"]).to_torch(dtype=pl.Int64).T
    edges["e_id"] = df_edges.select("edge_id").to_torch(dtype=pl.Int64).flatten()
    timestamps["edges"] = df_edges.select(["edge_id", "timestamp"])

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

    return Graph(nodes=nodes, edges=edges, timestamps=timestamps)

def load_graph(name: str) -> Graph:
    config = load_config(name)
    data = load_df(name)
    data = normalize_dataframes(data, config)
    return build_graph(data)

g = load_graph("yelp-zip")
