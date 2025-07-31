import logging

import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.data.dataset import load_df
from src.dgadb.models.SLADE.SLADE_main import SLADEModel
from src.dgadb.preprocessing.splitting import generate_data_splits
from src.dgadb.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

dataset = "bitcoin-alpha"

config = load_config(dataset)
window_size = config["window_size"]
train_ratio = config["train_ratio"]
val_ratio = config.get("val_ratio", None)

data = load_df(dataset)
data = generate_data_splits(
        data, train_ratio=train_ratio, val_ratio=val_ratio)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hyperparams = {}


model = SLADEModel(device, hyperparams, roc_auc_score)
model.setup(data["edges"].to_pandas())
model.train()
preds, labels, inf_time = model.inference("test")

# here we would add some evaluation metric function/class thingy
auc_score = roc_auc_score(labels, preds)
logger.info(f"Test ROC-AUC Score: {auc_score:.4f}")


