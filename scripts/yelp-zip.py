import polars as pl
import os

print("Yelp Zip")

base_path = os.environ["BASE_PATH"]
req_path = os.path.join(base_path, "raw/yelp-zip/yelpzip.csv")

if not os.path.exists(req_path):
    raise FileNotFoundError(
        f"Required file not found: {req_path}\n Download the required file at https://www.scidb.cn/en/detail?dataSetId=76673646fb7241f58aada7b9f24b25fe and place it in the raw folder."
    )
else:
    print("Raw file found!")

print("Processing...")

dataset_name = "yelp-zip"
dataset_dir = os.path.dirname(f"{base_path}/data/{dataset_name}/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

df = pl.read_csv(
    req_path,
    separator=",",
    skip_rows=1,
    has_header=False,
    columns=[0, 1, 2, 3, 4, 5, 6],
    new_columns=["edge_id", "src", "tgt", "rating", "label", "timestamp", "text"],
).sort(by="timestamp")

df_edges = df.select(["edge_id", "src", "tgt", "timestamp"]).with_columns(
    [pl.col("timestamp").str.strptime(pl.Date, "%Y-%m-%d").alias("timestamp")]
)
df_edge_features_num = (
    df.select(["edge_id", "rating"])
    .unpivot(["rating"], index=["edge_id"], variable_name="feature_id")
    .with_columns(pl.lit(0).cast(pl.Int64).alias("feature_id"))
)
df_edge_features_str = (
    df.select(["edge_id", "text"])
    .unpivot(["text"], index=["edge_id"], variable_name="feature_id")
    .with_columns(pl.lit(0).cast(pl.Int64).alias("feature_id"))
)

df_edge_labels = df.select(["edge_id", "label"]).with_columns(
    pl.when(pl.col("label") == -1).then(1).when(pl.col("label") == 1).then(0).otherwise(pl.col("label")).alias("label")
)

file_path_edges = os.path.join(dataset_dir, "edges.parquet")
df_edges.write_parquet(file_path_edges)
file_path_edge_labels = os.path.join(dataset_dir, "edge_labels.parquet")
df_edge_labels.write_parquet(file_path_edge_labels)
file_path_edge_features_num = os.path.join(dataset_dir, "edge_features_num.parquet")
df_edge_features_num.write_parquet(file_path_edge_features_num)
file_path_edge_features_str = os.path.join(dataset_dir, "edge_features_str.parquet")
df_edge_features_str.write_parquet(file_path_edge_features_str)


""" src_nodes = df.select([
    pl.col("src").alias("node_id"),
    pl.lit(0).alias("node_type")
])

tgt_nodes = df.select([
    pl.col("tgt").alias("node_id"),
    pl.lit(1).alias("node_type")
])

df_node_types = pl.concat([src_nodes, tgt_nodes]).unique()
file_path_node_types = os.path.join(dataset_dir, "node_types.parquet")
df_node_types.write_parquet(file_path_node_types) """

print("success.")
