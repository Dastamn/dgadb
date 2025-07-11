#!/usr/bin/env python3
import requests, gzip, io
import polars as pl
import os

print("Bitcoin")

base_path = os.environ["BASE_PATH"]

for t in ("alpha", "otc"):
    dataset_dir = os.path.dirname(f"{base_path}/data/bitcoin-{t}/")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print(f"Downloading Bitcoin-{t}...")
    url = f"https://snap.stanford.edu/data/soc-sign-bitcoin{t}.csv.gz"
    r = requests.get(url)
    r.raise_for_status()
    
    print(f"Processing Bitcoin-{t}...")
    with gzip.open(io.BytesIO(r.content), "rb") as f:
        df = pl.read_csv(
            f,
            has_header=False,
            separator=",",
            columns=[0,1,2,3],
            new_columns=["src","tgt","weight","timestamp"],
        )
        df = df.with_row_index(name="edge_id")

        # ids starting at 0
        df = df.with_columns([
            (pl.col("src") - 1).alias("src"),
            (pl.col("tgt") - 1).alias("tgt")
        ])
        
        df_edges = df.select(["edge_id", "src", "tgt", "timestamp"])
        df_edge_features_num = df.select(["edge_id", "weight"]).rename({"weight": "f0"}).unpivot(["f0"], index="edge_id", variable_name="feature_id")
        
    file_path_edges = os.path.join(dataset_dir, "edges.parquet")
    file_path_edge_features_num = os.path.join(dataset_dir, "edge_features_num.parquet")
    df_edges.write_parquet(file_path_edges)
    df_edge_features_num.write_parquet(file_path_edge_features_num)

print("success.")