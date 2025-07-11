import numpy as np
import polars as pl
import os

print("DGraph")
dataset_name = "dgraph"

base_path = os.environ["BASE_PATH"]
req_dir = os.path.dirname(f"{base_path}/raw/dgraph/")
dataset_dir = os.path.dirname(f"{base_path}/data/dgraph/")
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
            f"Required file not fouind: {file_path}\n Download the required files at https://dgraph.xinye.com and place them in the data/dgraph folder."
        )

print("Processing...")
graph = np.load(required_files[0])
node_time = np.load(required_files[1])
edge_time = np.load(required_files[2])

# there are two edge timestamps, we are using the one from the newer dataset

nodes = np.concatenate((graph["x"], node_time.reshape(-1, 1), graph["y"].reshape(-1, 1)), axis=1)

node_features = np.concatenate((node_time.reshape(-1, 1), graph["x"]), axis=1)
node_labels = graph["y"].reshape(-1, 1)

node_feature_cols = ["node_timestamp"] + ['f'+str(i) for i in range(17)]

df_node_features = pl.from_numpy(node_features, schema=node_feature_cols)
df_node_labels = pl.from_numpy(node_labels, schema=["label"])
df_node_features = df_node_features.with_row_index(name="node_id")
df_node_labels = df_node_labels.with_row_index(name="node_id")

df_node_features = df_node_features.unpivot([f"f{i}" for i in range(17)], index=["node_id"], variable_name="feature_id")


edges = np.concatenate((graph["edge_index"], edge_time.reshape(-1, 1)), axis=1)
df_edges = pl.from_numpy(edges, schema=["src", "tgt", "timestamp"])
df_edge_types = pl.from_numpy(graph["edge_type"].reshape(-1, 1), schema=["edge_type"])
df_edges = df_edges.with_row_index(name="edge_id")
df_edge_types = df_edge_types.with_row_index(name="edge_id")

file_path_node_features = os.path.join(dataset_dir, "node_features.parquet")
file_path_node_labels = os.path.join(dataset_dir, "node_labels.parquet")
file_path_edges = os.path.join(dataset_dir, "edges.parquet")
file_path_edge_types = os.path.join(dataset_dir, "edge_types.parquet")

df_node_features.write_parquet(file_path_node_features)
df_node_labels.write_parquet(file_path_node_labels)
df_edges.write_parquet(file_path_edges)
df_edge_types.write_parquet(file_path_edge_types)

print("success.")