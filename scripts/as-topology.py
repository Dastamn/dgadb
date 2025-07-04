#!/usr/bin/env python3
import requests, zipfile, io
import polars as pl
import os

dataset_name = "as-topology"

base_path = os.environ["BASE_PATH"]
dataset_dir = os.path.dirname(f"{base_path}/data/{dataset_name}/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url = "https://nrvis.com/download/data/dynamic/tech-as-topology.zip"
r = requests.get(url)
r.raise_for_status()

print("Processing...")
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    with z.open("tech-as-topology.edges") as f:
        df = pl.read_csv(
            f,
            separator=" ",
            has_header=False,
            skip_rows=5,
            columns=[0,1,3],
            new_columns=["src","tgt","timestamp"]
        )
df = df.with_row_index(name="edge_id")

# ids starting at 0
df = df.with_columns([
    (pl.col("src") - 1).alias("src"),
    (pl.col("tgt") - 1).alias("tgt")
])

file_path = os.path.join(dataset_dir, "edges.parquet")
df.write_parquet(file_path)

print("success.")
