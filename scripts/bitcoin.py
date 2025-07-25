#!/usr/bin/env python3
import requests
import gzip
import io
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
            columns=[0, 1, 2, 3],
            new_columns=["src", "tgt", "weight", "timestamp"],
        )
        df = df.with_row_index(name="edge_id")

        all_node_ids = pl.concat([df["src"], df["tgt"]]).unique().sort()
        id_map = {old_id: new_id for new_id,
                  old_id in enumerate(all_node_ids.to_list())}
        df = df.with_columns([
            pl.col("src").replace(id_map).alias("src"),
            pl.col("tgt").replace(id_map).alias("tgt"),
        ])
        print(df)
        df = df.sort(["timestamp", "weight"], descending=False)
        print(df)
        df_edges = df.select(["edge_id", "src", "tgt", "timestamp"])
        df_edge_features_num = df.select(["edge_id", "weight"]).rename(
            {"weight": "f0"}).unpivot(["f0"], index="edge_id", variable_name="feature_id")

        df_overall_state = df.group_by("tgt").agg(pl.sum("weight")).with_columns(pl.when(pl.col(
            "weight") >= 0).then(0).otherwise(1).alias("overall_state")).drop("weight").rename({"tgt": "node_id"})
        overall_state = set(df_overall_state.filter(
            pl.col("overall_state") == 1)["node_id"].to_list())
        current_dynamic_abnormal = set()

        edge_label_list = []
        for row in df.iter_rows(named=True):
            if row["weight"] < 0:
                # update the current dynamic state if needed
                if row["tgt"] in overall_state:
                    current_dynamic_abnormal.add(row["tgt"])
            if row["src"] in current_dynamic_abnormal:
                edge_label_list.append(1)
            else:
                edge_label_list.append(0)
        df_edge_labels = df.select(["edge_id"]).with_columns(
            pl.Series("label", edge_label_list)
        )

    file_path_edges = os.path.join(dataset_dir, "edges.parquet")
    file_path_edge_features_num = os.path.join(
        dataset_dir, "edge_features_num.parquet")
    file_path_edge_labels = os.path.join(
        dataset_dir, "edge_labels.parquet")
    df_edges.write_parquet(file_path_edges)
    df_edge_features_num.write_parquet(file_path_edge_features_num)
    df_edge_labels.write_parquet(file_path_edge_labels)

print("success.")
