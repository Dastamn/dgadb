import logging
import polars as pl
from sklearn.metrics import roc_auc_score

from src.dgadb.data.builder import build_graph
from src.dgadb.data.dataset import load_df
from src.dgadb.models.TADDY.TADDY_main import TADDYModel
from src.dgadb.preprocessing.snapshotting import assign_snapshots
from src.dgadb.utils import load_config
from src.dgadb.preprocessing.temporal import generate_data_splits
from dgadb.preprocessing.structural import make_undirected, remove_self_loops, remove_duplicates, reindex_nodes, remove_duplicates_train, remove_self_loops_train, make_undirected_train
from src.dgadb.preprocessing.anomaly_generation import AnomalyGenerator
from src.dgadb.preprocessing.temporal import normalize_timestamps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

dataset = "bitcoin-alpha"

config = load_config(dataset)
snapshot_size = config["snapshot_size"]
train_ratio = config["train_ratio"]
val_ratio = config.get("val_ratio", None)
anomaly_ratio = config.get("anomaly_ratio", 0.00)

meta_dict = {"dataset_name": dataset, "train_ratio": train_ratio,
             "val_ratio": val_ratio, "anomaly_ratio": anomaly_ratio}

data = load_df(dataset)
data["edges"] = generate_data_splits(data["edges"], train_ratio, val_ratio)
edge_features = data["edges"].select(["ff0_num"]).to_numpy()
data["edges"] = data["edges"].drop("label")
data["edges"] = normalize_timestamps(data["edges"])

ag = AnomalyGenerator(data["edges"], edge_features=edge_features)
data["edges"], edge_features = ag._generate_anomalous_samples(
    anomaly_ratio, "temporal")

data = make_undirected(data)
data = remove_self_loops(data)
data = remove_duplicates(data)
data = reindex_nodes(data)
prev_edges = len(data["edges"])

data = assign_snapshots(data, snapshot_size=snapshot_size)

# print(data["edges"]) # this is exactly as after 0_prepare_data.py in TADDY

device = "mps"
hyperparams = {}

model = TADDYModel(device, meta_dict, hyperparams,
                   epoch_evaluation_metric=roc_auc_score)

model.setup(data["edges"])
model.train()

preds, labels, inf_time = model.inference("test")
auc_full = roc_auc_score(labels, preds)
logger.info(f"TOTAL AUC: {auc_full:.4f}")
