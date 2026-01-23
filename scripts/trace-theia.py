#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "polars",
# ]
# ///
import zipfile
import io
import polars as pl
import os
import requests

print("Trace & Theia")
dataset_name = "dgraph"

data_path = os.environ["DATA_PATH"]


print("Downloading...")
url = "https://github.com/JakubReha/ProvCTDG/releases/download/v1.0.0.0/DATA.zip"

r = requests.get(url)
r.raise_for_status()

print("Processing...")

datasets = {"theia": "darpa_theia_0to24", "trace": "darpa_trace_0to210"}

for ds in datasets.keys():
    print(f"  -{ds}")
    dataset_dir = os.path.dirname(f"{data_path}/data/{ds}/")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(f"DATA/{datasets[ds]}/edges.csv") as f:
            df = pl.read_csv(
                f,
                skip_lines=1,
                separator=",",
                has_header=False,
                columns=[0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14],
                new_columns=["edge_id", "src", "tgt", "f0", "f1",
                             "f2", "timestamp", "f3", "f4", "f5", "f6", "label"]
            )
        with z.open(f"DATA/{datasets[ds]}/attributed_nodes.csv") as f:
            df_n = pl.read_csv(
                f,
                skip_lines=1,
                separator=",",
                has_header=False,
                columns=[0, 2, 4, 5, 6, 7, 8],
                new_columns=["node_id", "f0_str", "type",
                             "f1_str", "f0_cat", "f2_str", "f1_cat"]
            )

        df_edges = df.select(["edge_id", "src", "tgt", "timestamp"])
        df_edge_labels = df.select(["edge_id", "label"]).with_columns(
            pl.col("label").cast(pl.Int32)
        )
        df_edge_features_num = df.drop(["src", "tgt", "timestamp", "label"]).unpivot(
            [f"f{i}" for i in range(7)],
            index=["edge_id"],
            variable_name="feature_id"
        ).with_columns(pl.col("feature_id").str.extract(r"(\d+)").cast(pl.Int64))

        file_path_edges = os.path.join(dataset_dir, "edges.parquet")
        df_edges.write_parquet(file_path_edges)
        file_path_edge_labels = os.path.join(
            dataset_dir, "edge_labels.parquet")
        df_edge_labels.write_parquet(file_path_edge_labels)
        file_path_edge_features_num = os.path.join(
            dataset_dir, "edge_features_num.parquet")
        df_edge_features_num.write_parquet(file_path_edge_features_num)

        del df, df_edges, df_edge_labels, df_edge_features_num

        df_node_types = df_n.select(["node_id", "type"])

        df_node_attributes_cat = df_n.select(["node_id", "f0_cat", "f1_cat"]).unpivot(
            [f"f{i}_cat" for i in range(2)],
            index=["node_id"],
            variable_name="feature_id"
        ).with_columns(pl.col("feature_id").str.extract(r"(\d+)").cast(pl.Int64))

        df_node_attributes_str = df_n.select(["node_id", "f0_str", "f1_str", "f2_str"]).unpivot(
            [f"f{i}_str" for i in range(3)],
            index=["node_id"],
            variable_name="feature_id"
        ).with_columns(pl.col("feature_id").str.extract(r"(\d+)").cast(pl.Int64))

        file_path_node_types = os.path.join(dataset_dir, "node_types.parquet")
        df_node_types.write_parquet(file_path_node_types)
        file_path_node_attributes_cat = os.path.join(
            dataset_dir, "node_attributes_cat.parquet")
        df_node_attributes_cat.write_parquet(file_path_node_attributes_cat)
        file_path_node_attributes_str = os.path.join(
            dataset_dir, "node_attributes_str.parquet")
        df_node_attributes_str.write_parquet(file_path_node_attributes_str)

        del df_n, df_node_attributes_cat, df_node_attributes_str, df_node_types
        print(f"  -done")
print("success.")
