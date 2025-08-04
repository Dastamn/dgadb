from ..data.dataset import load_df
from ..storage import Graph
from ..preprocessing.snapshotting import assign_snapshots
from ..preprocessing.temporal import normalize_timestamps, generate_data_splits
from ..preprocessing.normalization import get_normalized_feature_matrices
from ..data.builder import build_graph
from ..utils import load_config


def load_graph(name: str) -> Graph:
    config = load_config(name)
    window_size = config["window_size"]
    train_ratio = config["train_ratio"]
    val_ratio = config.get("val_ratio", None)

    data = load_df(name)

    edges = data["edges"]
    edges = normalize_timestamps(edges)
    edges = generate_data_splits(
        edges, train_ratio=train_ratio, val_ratio=val_ratio)

    data["edges"] = edges

    node_features, edge_features = get_normalized_feature_matrices(
        data["nodes"], data["edges"], config)

    data = assign_snapshots(data, window_size=window_size)

    # snapshot_split_map = generate_data_splits(
    #     data, train_ratio=train_ratio, val_ratio=val_ratio)

    # data["edges"] = data["edges"].join(snapshot_split_map, on="snapshot_id")
    # if "nodes" in data and "snapshot_id" in data["nodes"].columns:
    #     data["nodes"] = data["nodes"].join(
    #         snapshot_split_map, on="snapshot_id")

    # data = normalize_dataframes(data, config)

    return build_graph(data, node_features, edge_features, window_size)
