#!/usr/bin/env python3
import requests
import tarfile
import io
import polars as pl
import os

print("UC-Social")

data_path = os.environ["DATA_PATH"]
dataset_dir = os.path.dirname(f"{data_path}/data/uc-social/")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("Downloading...")
url = "http://konect.cc/files/download.tsv.opsahl-ucsocial.tar.bz2"
# asym positive
r = requests.get(url)
r.raise_for_status()

print("Processing...")
with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:bz2") as tar:
    member = tar.getmember("opsahl-ucsocial/out.opsahl-ucsocial")
    with tar.extractfile(member) as f:
        df = pl.read_csv(
            f, skip_rows=2, separator=" ", has_header=False, columns=[0, 1, 3], new_columns=["src", "tgt", "timestamp"]
        ).sort(by="timestamp")
df = df.with_row_index(name="edge_id")

# ids starting at 0
df = df.with_columns([(pl.col("src") - 1).alias("src"), (pl.col("tgt") - 1).alias("tgt")])
print(df)
file_path = os.path.join(dataset_dir, "edges.parquet")
df.write_parquet(file_path)

print("success.")
