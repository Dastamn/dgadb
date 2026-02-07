#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "polars",
# ]
# ///
import requests
import zipfile
import io
import polars as pl
import os

print("Reddit")
dataset_name = "reddit"

data_path = os.environ["DATA_PATH"]
dataset_dir = os.path.dirname(f"{data_path}/data/{dataset_name}/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url = "http://snap.stanford.edu/jodie/reddit.csv"
r = requests.get(url)
r.raise_for_status()

print("Processing...")
buf = io.BytesIO(r.content)

df = pl.read_csv(
    buf,
    separator=",",
    skip_lines=1,
    has_header=False,
    null_values=["-"],
    new_columns=["src", "tgt", "timestamp", "label"] + [f"f{i}" for i in range(172)],
).sort(by="timestamp")
df = df.with_row_index(name="edge_id")

# one set of ids for all types
n_src = df["src"].max() + 1
df = df.with_columns((pl.col("tgt") + n_src).alias("tgt"))

df_edges = df.select(["edge_id", "src", "tgt", "timestamp"])
df_edge_labels = df.select(["edge_id", "label"])
df_edge_features = df.drop(["src", "tgt", "timestamp", "label"]).unpivot(
    [f"f{i}" for i in range(172)], index=["edge_id"], variable_name="feature_id"
)
df_edge_features = df_edge_features.with_columns(pl.col("feature_id").str.extract(r"(\d+)").cast(pl.Int64))

# add node types
total_nodes = df["tgt"].max() + 1
df_node_types = pl.DataFrame({"node_id": range(total_nodes)})
df_node_types = df_node_types.with_columns(pl.when(pl.col("node_id") < n_src).then(0).otherwise(1).alias("node_type"))

file_path_edges = os.path.join(dataset_dir, "edges.parquet")
df_edges.write_parquet(file_path_edges)

file_path_edge_labels = os.path.join(dataset_dir, "edge_labels.parquet")
df_edge_labels.write_parquet(file_path_edge_labels)

file_path_edge_features = os.path.join(dataset_dir, "edge_features_num.parquet")
df_edge_features.write_parquet(file_path_edge_features)

file_path_node_types = os.path.join(dataset_dir, "node_types.parquet")
df_node_types.write_parquet(file_path_node_types)


print("success.")
