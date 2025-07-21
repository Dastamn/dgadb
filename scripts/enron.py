#!/usr/bin/env python3
import requests
import tarfile
import io
import polars as pl
import os

print("Enron")
dataset_name = "enron"

base_path = os.environ["BASE_PATH"]
dataset_dir = os.path.dirname(f"{base_path}/data/{dataset_name}/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url = "http://konect.cc/files/download.tsv.enron.tar.bz2"
r = requests.get(url)
r.raise_for_status()

print("Processing...")
with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:bz2") as tar:
    member = tar.getmember("enron/out.enron")
    with tar.extractfile(member) as f:
        df = pl.read_csv(
            f,
            skip_rows=1,
            separator=" ",
            has_header=False,
            columns=[0, 1, 3],
            new_columns=["src", "tgt", "timestamp"]
        )
        df = df.with_row_index(name="edge_id")

file_path = os.path.join(dataset_dir, "edges.parquet")
df.write_parquet(file_path)

print("success.")
