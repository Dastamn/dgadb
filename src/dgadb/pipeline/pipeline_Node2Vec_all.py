from dgadb.models.baseline.Node2Vec import n2vModel

import argparse
import logging
import torch
from sklearn.metrics import roc_auc_score
from dgadb.storage import TemporalGraphLoader, TemporalGraph
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

datasets = [
    "bitcoin-alpha",
    "bitcoin-otc",
    "uc-social",
    "digg-homo",
    "as-topology",
    "email-dnc",
    "enron",
    "epinions",
    "mooc",
    "reddit",
    "dgraph",
    "wiki"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNNBaseline pipeline all datasets")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--snap_size", type=int, default=2000)

    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in datasets:
        raise RuntimeError(f"Unknown dataset: '{dataset_name}'.")

    device = torch.device("cpu")

    loader = TemporalGraphLoader()
    tg = loader.load(
        dataset_name,
        anom_type="s",
        anom_test_ratio=0.1,
        anom_val_ratio=0.1,
        create_if_not_found=True
    )

    hyperparams = {
        "num_epochs": args.epochs,
        "snap_size": args.snap_size,
    }

    model = n2vModel(device, {}, hyperparams, roc_auc_score)
    model.setup(tg)
    model.train()
    probs, split_labels = model.inference(split="test")
    print("TOTAL AUC")
    print(roc_auc_score(split_labels.detach().numpy(), probs.detach().numpy()))
