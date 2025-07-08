import polars as pl
import os
import requests
import io
import gzip

print("Amazon")

base_path = os.environ["BASE_PATH"]

# TODO create labels from fraction of helpful-to-total votes anomaly if <0.25, paper filters out >0.75 as well

ds = {
    #"am":"Automotive", cannot run on my computer
    "bp":"Baby_Products",
    "mi":"Musical_Instruments"
}

for product in ds.keys():
    print(f"Amazon: {ds[product]}")
    
    dataset_dir = os.path.dirname(f"{base_path}/data/amazon_{product}/")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print(" - Downloading reviews...")
    url_review = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{ds[product]}.jsonl.gz"

    r = requests.get(url_review)
    r.raise_for_status()
    buf = io.BytesIO(r.content)

    with gzip.GzipFile(fileobj=buf, mode="rb") as gzfile:
        df = pl.read_ndjson(gzfile).drop(["images"])

    print(" - processing reviews...")

    unique_ids = pl.concat([df["user_id"], df["parent_asin"]]).unique().to_frame("str_id")
    unique_ids = unique_ids.with_row_index(name="int_id")

    df = df.join(unique_ids, left_on="user_id", right_on="str_id").drop(["user_id"]).rename({"int_id":"src"})
    df = df.join(unique_ids, left_on="parent_asin", right_on="str_id").drop(["parent_asin"]).rename({"int_id":"tgt"})
    df= df.with_row_index(name="edge_id")

    df_edges = df.select(["edge_id", "src", "tgt", "timestamp"])
    
    df_src_nodes = (
        df_edges.select(pl.col("src").alias("node_id"))
        .unique()
        .with_columns(pl.lit(0).alias("node_type"))
    )

    df_tgt_nodes = (
        df_edges.select(pl.col("tgt").alias("node_id"))
        .unique()
        .with_columns(pl.lit(1).alias("node_type"))
    )

    df_node_types = pl.concat([df_src_nodes, df_tgt_nodes])
    del df_src_nodes, df_tgt_nodes

    file_path_edges = os.path.join(f"{base_path}/data/amazon_{product}/", "edges.parquet")
    df_edges.write_parquet(file_path_edges)
    del df_edges

    file_path_node_types = os.path.join(f"{base_path}/data/amazon_{product}/", "node_types.parquet")
    df_node_types.write_parquet(file_path_node_types)
    del df_node_types

    df_edge_features_cat = df.select(["edge_id", "verified_purchase"])
    df_edge_features_num = df.select(["edge_id", "helpful_vote", "rating"]) #is rating cat or num?
    df_edge_features_str = df.select(["edge_id", "title", "text"])

    edge_features = {
        "cat": df_edge_features_cat,
        "num": df_edge_features_num,
        "str": df_edge_features_str,
    }

    for t in ("cat", "num", "str"):
        file_path = os.path.join(f"{base_path}/data/amazon_{product}/", f"edge_features_{t}.parquet")
        edge_features[t].write_parquet(file_path)
        del edge_features[t]

    del df
    del edge_features



    schema = {
        "main_category": pl.Utf8,
        "title": pl.Utf8,
        "average_rating": pl.Float64,
        "rating_number": pl.Int64,
        "features": pl.List(pl.Utf8),
        "description": pl.List(pl.Utf8),
        "images": pl.Utf8,
        "videos": pl.Utf8,
        "price": pl.Utf8,
        "store": pl.Utf8,
        "categories": pl.List(pl.Utf8),
        "details": pl.Utf8,
        "parent_asin": pl.Utf8,
        "bought_together": pl.Object,
    }

    print(" - downloading users...")
    url_meta = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{ds[product]}.jsonl.gz"

    r = requests.get(url_meta)
    r.raise_for_status()
    buf = io.BytesIO(r.content)

    with gzip.GzipFile(fileobj=buf, mode="rb") as gzfile:
        df = pl.read_ndjson(gzfile, schema=schema)

    print(" - processing users...")
    print("")
   
    df = df.drop(["bought_together", "images", "videos"])
    df = df.join(unique_ids, left_on="parent_asin", right_on="str_id").drop(["parent_asin"]).rename({"int_id":"node_id"})

    df_node_features_str = df.select(["node_id", "main_category", "title", "features", "description", "store", "categories", "details"])
    df_node_features_num = df.select(["node_id", "average_rating", "rating_number", "price"]).rename({"average_rating":"f0", "rating_number":"f1", "price":"f2"}).unpivot([f"f{i}" for i in range(3)],
        index = ["node_id"],
        variable_name="feature_id"
    ).with_columns(pl.col("feature_id").str.extract(r"(\d+)").cast(pl.Int64))
    # TODO need to unpivot str as well, not sure how to deal with list[str] and details col

    node_features = {
        "num": df_node_features_num,
        "str": df_node_features_str
    }
    for t in ("num", "str"):
        file_path = os.path.join(f"{base_path}/data/amazon_{product}/", f"node_features_{t}.parquet")
        node_features[t].write_parquet(file_path)
        del node_features[t]

    del df
    del node_features

print("success.")
