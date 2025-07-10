#!/usr/bin/env python3
import requests, gzip, io
import polars as pl
import os

print("Wiki")

base_path = os.environ["BASE_PATH"]
# overlapping ids for src and target, need to fix!

dataset_dir = os.path.dirname(f"{base_path}/data/wiki/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url="http://snap.stanford.edu/jodie/wikipedia.csv"
r = requests.get(url)
r.raise_for_status()

print(f"Processing...")
feature_cols = [f"f{i}" for i in range(10)]
df = pl.read_csv(
    io.BytesIO(r.content),
    has_header=False,
    separator=",",
    skip_rows=1,
    new_columns=["src","tgt","timestamp", "label"] + [f"f{i}" for i in range(10)]
    )
df = df.with_row_index("edge_id")

# one set of ids for all types
n_src = df["src"].max() + 1
df = df.with_columns(
    (pl.col("tgt") + n_src).alias("tgt")
)

df_edges = df.select(["edge_id", "src", "tgt", "timestamp"])
df_edge_labels = df.select(["edge_id", "label"])
df_edge_features_num = df.select(["edge_id"] + feature_cols)
df_edge_features_num = df_edge_features_num.unpivot(feature_cols, index=["edge_id"], variable_name="feature_id")

# add node types
total_nodes = df_edges["tgt"].max() + 1
df_node_types = pl.DataFrame({"node_id": range(total_nodes)})
df_node_types = df_node_types.with_columns(
    pl.when(pl.col("node_id") < n_src)
    .then(0)
    .otherwise(1)
    .alias("node_type")
)

file_path_edges = os.path.join(dataset_dir, "edges.parquet")
file_path_edge_labels = os.path.join(dataset_dir, "edge_labels.parquet")
file_path_edge_features_num = os.path.join(dataset_dir, "edge_features_num.parquet")
file_path_node_types = os.path.join(dataset_dir, "node_types.parquet")
df_edges.write_parquet(file_path_edges)
df_edge_labels.write_parquet(file_path_edge_labels)
df_edge_features_num.write_parquet(file_path_edge_features_num)
df_edge_features_num.write_parquet(file_path_node_types)

print("success.")