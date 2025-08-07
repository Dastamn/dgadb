#!/usr/bin/env python3
import requests
import tarfile
import io
import polars as pl
import os

print("Epinions")
dataset_name = "epinions"

base_path = os.environ["BASE_PATH"]
dataset_dir = os.path.dirname(f"{base_path}/data/{dataset_name}/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url = "http://konect.cc/files/download.tsv.epinions-rating.tar.bz2"
r = requests.get(url)
r.raise_for_status()

print("Processing...")
with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:bz2") as tar:
    member = tar.getmember("epinions-rating/out.epinions-rating")
    with tar.extractfile(member) as f:
        df = pl.read_csv(
            f,
            skip_rows=1,
            separator=" ",
            has_header=False,
            columns=[0, 1, 2, 3],
            new_columns=["src", "tgt", "feature", "timestamp"],
        ).sort(by="timestamp")
df = df.with_row_index(name="edge_id")

# ids starting from 0
df = df.with_columns([(pl.col("src") - 1).alias("src"), (pl.col("tgt") - 1).alias("tgt")])

# one set of ids for all types
n_src = df["src"].max() + 1
df = df.with_columns((pl.col("tgt") + n_src).alias("tgt"))

""" # add node types
total_nodes = df["tgt"].max() + 1
df_node_types = pl.DataFrame({"node_id": range(total_nodes)})
df_node_types = df_node_types.with_columns(
    pl.when(pl.col("node_id") < n_src)
      .then(0)
      .otherwise(1)
      .alias("node_type")
) """

df_edges = df.select(["edge_id", "src", "tgt", "timestamp"])
df_edge_features = (
    df.select(["edge_id", "feature"])
    .rename({"feature": "value"})
    .with_columns(pl.lit(0).alias("feature_id"))
    .select(["edge_id", "feature_id", "value"])
)

file_path_edges = os.path.join(dataset_dir, "edges.parquet")
file_path_edge_features = os.path.join(dataset_dir, "edge_features.parquet")
# file_path_node_types = os.path.join(dataset_dir, "node_types.parquet")
df_edges.write_parquet(file_path_edges)
df_edge_features.write_parquet(file_path_edge_features)
# df_node_types.write_parquet(file_path_node_types)

print("success.")
