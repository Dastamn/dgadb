import logging
import polars as pl
from sklearn.metrics import roc_auc_score
import numpy as np
from src.dgadb.data.builder import build_graph
from src.dgadb.data.dataset import load_df
from src.dgadb.models.StrGNN.StrGNN_main import STRGNNModel
from src.dgadb.preprocessing.snapshotting import assign_snapshots
from src.dgadb.utils import load_config
from src.dgadb.preprocessing.temporal import generate_data_splits
from src.dgadb.preprocessing.structural import (
    make_undirected,
    remove_self_loops,
    remove_duplicates,
    reindex_nodes,
    remove_duplicates_train,
    remove_self_loops_train,
    make_undirected_train,
)
from src.dgadb.preprocessing.anomaly_generation import AnomalyGenerator
from src.dgadb.preprocessing.temporal import normalize_timestamps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

dataset = "bitcoin-otc"

config = load_config(dataset)
snapshot_size = config["snapshot_size"]
train_ratio = config["train_ratio"]
val_ratio = config.get("val_ratio", 0.0)
anomaly_ratio = config.get("anomaly_ratio", 0.01)

meta_dict = {
    "dataset_name": dataset,
    "train_ratio": train_ratio,
    "val_ratio": val_ratio,
    "anomaly_ratio": anomaly_ratio,
}

data = load_df(dataset)
data["edges"] = generate_data_splits(data["edges"], train_ratio)
edge_features = np.ones(((len(data["edges"]), 1)))
if not (dataset == "bitcoin-alpha" or dataset == "bitcoin-otc"):
    # if "label" in data["edges"].columns:
    #    data["edges"] = data["edges"].drop("label")

    data["edges"] = normalize_timestamps(data["edges"])

    # ag = AnomalyGenerator(data["edges"], edge_features=edge_features)
    # data["edges"], edge_features = ag._generate_anomalous_samples(anomaly_ratio, "structural", temporal_window_size=50)
    # data["edges"], edge_features = ag._generate_anomalous_samples(anomaly_ratio, "structural", temporal_window_size=50, use_val_split=True)


data = assign_snapshots(data, snapshot_size=snapshot_size)

# print(data["edges"]) # this is exactly as after 0_prepare_data.py in TADDY

device = "mps"
hyperparams = {}

model = STRGNNModel(device, meta_dict, hyperparams, epoch_evaluation_metric=roc_auc_score)

model.setup(data["edges"])
model.train()
preds, labels, inf_time = model.inference("test")
auc_full = roc_auc_score(labels, preds)
logger.info(f"Total auc on test: {auc_full:.4f}")
# TODO: Make it possible to run datasets with their original labels (perhaps together with negative sampling)
