import argparse

import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.baseline import GNNBaseline, GNN_BASELINES_DICT
from src.dgadb.storage import TemporalGraphLoader, TemporalGraph


datasets = ["bitcoin-alpha"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNNBaseline pipeline all datasets")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str,
                        choices=GNN_BASELINES_DICT.keys(), default="GCN")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--snap_size", type=int, default=2000)

    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in datasets:
        raise RuntimeError(f"Unknown dataset: '{dataset_name}'.")

    device = torch.device("cpu")

    loader = TemporalGraphLoader()
    tg = loader.load(dataset_name, anom_type="s", anom_test_ratio=0.1,
                     anom_val_ratio=0.1, create_if_not_found=True)

    hyperparams = {"num_epoch": args.epochs,
                   "snap_size": args.snap_size, "model": args.model}

    model = GNNBaseline(device, {}, hyperparams, roc_auc_score)
    model.setup(tg)
    model.train()
