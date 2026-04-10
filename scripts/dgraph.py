#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "polars",
# ]
# ///
import numpy as np
import polars as pl
import os

print("DGraph")
dataset_name = "dgraph"

data_path = os.environ["DATA_PATH"]
req_dir = os.path.dirname(f"{data_path}/raw/dgraph/")
dataset_dir = os.path.dirname(f"{data_path}/data/dgraph/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Looking for required files...")

required_files = [
    os.path.join(req_dir, "dgraphfin.npz"),
    os.path.join(req_dir, "dgraphfinv2_node_timestamp.npy"),
    os.path.join(req_dir, "dgraphfinv2_edge_timestamp.npy")
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n Download the required files at https://dgraph.xinye.com and place them in the data/dgraph folder."
        )

print("Processing...")
graph = np.load(required_files[0])
node_time = np.load(required_files[1])
edge_time = np.load(required_files[2])

# there are two edge timestamps, we are using the one from the newer dataset

nodes = np.concatenate(
    (graph["x"], node_time.reshape(-1, 1), graph["y"].reshape(-1, 1)), axis=1)

node_features_num = graph["x"]
node_time = node_time.reshape(-1, 1)
node_labels = graph["y"].reshape(-1, 1)

node_feature_cols = ['f'+str(i) for i in range(17)]

df_node_features_num = pl.from_numpy(
    node_features_num, schema=node_feature_cols)
df_node_labels = pl.from_numpy(node_labels, schema=["label"])
df_node_timestamps = pl.from_numpy(node_time, schema=["timestamp"])

df_node_features_num = df_node_features_num.with_row_index(name="node_id")
df_node_labels = df_node_labels.with_row_index(name="node_id")
df_node_timestamps = df_node_timestamps.with_row_index(name="node_id")

df_node_features_num = df_node_features_num.unpivot(
    [f"f{i}" for i in range(17)], index=["node_id"], variable_name="feature_id")

edges = np.concatenate((graph["edge_index"], edge_time.reshape(-1, 1)), axis=1)
df_edges = pl.from_numpy(edges, schema=["src", "tgt", "timestamp"])
df_edge_types = pl.from_numpy(
    graph["edge_type"].reshape(-1, 1), schema=["edge_type"])
df_edges = df_edges.with_row_index(name="edge_id")
df_edge_types = df_edge_types.with_row_index(name="edge_id")

# Derive edge labels from node labels:
# edge is anomalous (1) if either endpoint node has label 1, else normal (0)
df_src_labels = df_node_labels.rename({"node_id": "src", "label": "src_label"})
df_tgt_labels = df_node_labels.rename({"node_id": "tgt", "label": "tgt_label"})

df_edge_labels = (
    df_edges
    .join(df_src_labels, on="src", how="left")
    .join(df_tgt_labels, on="tgt", how="left")
    .with_columns(
        pl.when((pl.col("src_label") == 1) | (pl.col("tgt_label") == 1))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("label")
    )
    .select(["edge_id", "label"])
)

file_path_node_features_num = os.path.join(
    dataset_dir, "node_features_num.parquet")
file_path_node_labels = os.path.join(dataset_dir, "node_labels.parquet")
file_path_node_timestamps = os.path.join(
    dataset_dir, "node_timestamps.parquet")
file_path_edges = os.path.join(dataset_dir, "edges.parquet")
file_path_edge_types = os.path.join(dataset_dir, "edge_types.parquet")
file_path_edge_labels = os.path.join(dataset_dir, "edge_labels.parquet")

df_node_features_num.write_parquet(file_path_node_features_num)
df_node_labels.write_parquet(file_path_node_labels)
df_node_timestamps.write_parquet(file_path_node_timestamps)
df_edges.write_parquet(file_path_edges)
df_edge_types.write_parquet(file_path_edge_types)
df_edge_labels.write_parquet(file_path_edge_labels)

print("success.")
