#!/usr/bin/env python3
import requests
import tarfile
import io
import polars as pl
import os

print("MOOC")
dataset_name = "mooc"

base_path = os.environ["BASE_PATH"]
dataset_dir = os.path.dirname(f"{base_path}/data/{dataset_name}/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url = "https://snap.stanford.edu/data/act-mooc.tar.gz"
r = requests.get(url)
r.raise_for_status()

print("Processing...")
with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
    actions = tar.getmember("act-mooc/mooc_actions.tsv")
    with tar.extractfile(actions) as f:
        df_edges = pl.read_csv(
            f,
            separator="\t",
            has_header=True,
            columns=[0, 1, 2, 3],
        )
        df_edges = df_edges.rename(
            {"ACTIONID": "edge_id", "USERID": "src", "TARGETID": "tgt", "TIMESTAMP": "timestamp"}
        )
        # one set of ids for all types
        n_src = df_edges["src"].n_unique()
        df_edges = df_edges.with_columns((pl.col("tgt") + n_src).alias("tgt")).sort(by="timestamp")

        """ # add node types
        total_nodes = df_edges["tgt"].max() + 1
        df_node_types = pl.DataFrame({"node_id": range(total_nodes)})
        df_node_types = df_node_types.with_columns(
            pl.when(pl.col("node_id") < n_src)
            .then(0)
            .otherwise(1)
            .alias("node_type")
        ) """

    features = tar.getmember("act-mooc/mooc_action_features.tsv")
    with tar.extractfile(features) as f:
        df_edge_features = pl.read_csv(
            f,
            separator="\t",
            has_header=True,
            columns=[0, 1, 2, 3, 4],
        )
        df_edge_features = df_edge_features.unpivot(
            ["FEATURE0", "FEATURE1", "FEATURE2", "FEATURE3"], index=["ACTIONID"], variable_name="feature_id"
        )

        df_edge_features = df_edge_features.with_columns(pl.col("feature_id").str.extract(r"(\d+)").cast(pl.Int64))

        df_edge_features = df_edge_features.rename({"ACTIONID": "edge_id"})

    labels = tar.getmember("act-mooc/mooc_action_labels.tsv")
    with tar.extractfile(labels) as f:
        df_edge_labels = pl.read_csv(
            f,
            separator="\t",
            has_header=True,
            columns=[0, 1],
        )
        df_edge_labels = df_edge_labels.rename({"ACTIONID": "edge_id", "LABEL": "label"})

file_path_edges = os.path.join(dataset_dir, "edges.parquet")
df_edges.write_parquet(file_path_edges)

file_path_edge_features = os.path.join(dataset_dir, "edge_features.parquet")
df_edge_features.write_parquet(file_path_edge_features)

file_path_edge_labels = os.path.join(dataset_dir, "edge_labels.parquet")
df_edge_labels.write_parquet(file_path_edge_labels)

# file_path_node_types = os.path.join(dataset_dir, "node_types.parquet")
# df_node_types.write_parquet(file_path_node_types)

print("success.")
