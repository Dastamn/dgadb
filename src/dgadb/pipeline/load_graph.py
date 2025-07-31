from ..data.dataset import load_df
from ..storage import Graph
from ..preprocessing.snapshotting import assign_snapshots
from ..preprocessing.splitting import generate_data_splits
from ..preprocessing.normalization import normalize_dataframes
from ..data.builder import build_graph
from ..utils import load_config


def load_graph(name: str) -> Graph:
    config = load_config(name)
    window_size = config["window_size"]
    train_ratio = config["train_ratio"]
    val_ratio = config.get("val_ratio", None)

    data = load_df(name)
    data = assign_snapshots(data, window_size=window_size)

    snapshot_split_map = generate_data_splits(
        data, train_ratio=train_ratio, val_ratio=val_ratio)

    data["edges"] = data["edges"].join(snapshot_split_map, on="snapshot_id")
    if "nodes" in data and "snapshot_id" in data["nodes"].columns:
        data["nodes"] = data["nodes"].join(
            snapshot_split_map, on="snapshot_id")

    data = normalize_dataframes(data, config)
    return build_graph(data, snapshot_split_map=snapshot_split_map)