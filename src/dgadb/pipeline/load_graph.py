from src.dgadb.data.dataset import load_df
from src.dgadb.storage import Graph
from src.dgadb.preprocessing.snapshotting import assign_snapshots
from src.dgadb.preprocessing.temporal import normalize_timestamps, generate_data_splits
from src.dgadb.preprocessing.normalization import get_normalized_feature_matrices
from src.dgadb.data.builder import build_graph
from src.dgadb.utils import load_config


def load_graph(name: str) -> Graph:
    config = load_config(name)
    snapshot_size = config.get("snapshot_size", config.get("window_size", 1000))
    train_ratio = config["train_ratio"]
    val_ratio = config.get("val_ratio", None)

    data = load_df(name)

    edges = data["edges"]
    edges = normalize_timestamps(edges)
    edges = generate_data_splits(
        edges, train_ratio=train_ratio, val_ratio=val_ratio)

    data["edges"] = edges

    node_features, edge_features = get_normalized_feature_matrices(
        data.get("nodes"), data["edges"], config)

    data = assign_snapshots(data, snapshot_size=snapshot_size)

    # snapshot_split_map = generate_data_splits(
    #     data, train_ratio=train_ratio, val_ratio=val_ratio)

    # data["edges"] = data["edges"].join(snapshot_split_map, on="snapshot_id")
    # if "nodes" in data and "snapshot_id" in data["nodes"].columns:
    #     data["nodes"] = data["nodes"].join(
    #         snapshot_split_map, on="snapshot_id")

    # data = normalize_dataframes(data, config)

    return build_graph(data, node_features, edge_features, snapshot_size)
