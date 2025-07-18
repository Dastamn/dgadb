import polars as pl
from typing import Dict
import os

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