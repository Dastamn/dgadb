#!/usr/bin/env python3
import requests
import zipfile
import io
import polars as pl
import os

print("Email-DNC")
dataset_name = "email-dnc"

data_path = os.environ["DATA_PATH"]
dataset_dir = os.path.dirname(f"{data_path}/data/{dataset_name}/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url = "https://nrvis.com/download/data/dynamic/email-dnc.zip"
r = requests.get(url)
r.raise_for_status()

print("Processing...")
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    with z.open("email-dnc.edges") as f:
        df = pl.read_csv(f, separator=",", has_header=False, columns=[0, 1, 2], new_columns=["src", "tgt", "timestamp"])
        df = df.with_row_index(name="edge_id")

# ids starting at 0
df = df.with_columns([(pl.col("src") - 1).alias("src"), (pl.col("tgt") - 1).alias("tgt")])
file_path = os.path.join(dataset_dir, "edges.parquet")
df.write_parquet(file_path)

print("success.")
