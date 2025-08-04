import logging
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.data.builder import build_graph
from src.dgadb.data.dataset import load_df
from src.dgadb.models.TADDY.TADDY_main import TADDYModel
from src.dgadb.preprocessing.snapshotting import assign_snapshots
from src.dgadb.utils import load_config
from src.dgadb.preprocessing.temporal import generate_data_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

dataset = "bitcoin-alpha"

config = load_config(dataset)
window_size = config["window_size"]
train_ratio = config["train_ratio"]
val_ratio = config.get("val_ratio", None)
data = load_df(dataset)
data["edges"] = generate_data_splits(data["edges"], train_ratio, val_ratio)
print(data)
edge_features = data["edges"].select(["ff0_num"]).to_numpy()
data = assign_snapshots(data, window_size=window_size)
g = build_graph(
    data, node_features=None, edge_features=edge_features, window_size=window_size
)  # wont work, fix when splitting done

device = "cpu"
hyperparams = {}

model = TADDYModel(device, hyperparams, epoch_evaluation_metric=roc_auc_score)
model.setup(g)
model.train()
preds, labels, inf_time = model.inference("test")
