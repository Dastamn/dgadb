import logging
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from src.dgadb.preprocessing.temporal import normalize_timestamps
import numpy as np
from src.dgadb.data.dataset import load_df
from src.dgadb.models.GeneralDYG.GeneralDYG_main import GeneralDYGModel
from src.dgadb.preprocessing.temporal import generate_data_splits
from src.dgadb.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

dataset = "bitcoin-alpha"

config = load_config(dataset)
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
data["edges"] = generate_data_splits(data["edges"], train_ratio=train_ratio, val_ratio=val_ratio)

data["edges"] = normalize_timestamps(data["edges"])
edge_features = np.ones((len(data["edges"]), 1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hyperparams = {}

model = GeneralDYGModel(device, meta_dict, hyperparams, roc_auc_score)
model.setup(data["edges"])
model.train()
preds, labels, inf_time = model.inference("test")

# here we would add some evaluation metric function/class thingy
auc_score = roc_auc_score(labels, preds)
logger.info(f"Test ROC-AUC Score: {auc_score:.4f}")
