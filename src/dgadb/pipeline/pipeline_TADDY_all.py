import argparse
from time import time
import logging
from src.dgadb.preprocessing.pipeline import Pipeline
from src.dgadb.preprocessing.anomaly_injector import AnomalyInjector
from src.dgadb.preprocessing.add_graph_temporary import inject_anomalies_addgraph_style
from src.dgadb.data.builder import build_graph_from_temporal
from src.dgadb.models.RustGraph.main_RustGraph import RustGraphModel
from src.dgadb.evaluation.evaluator import Evaluator
from src.dgadb.storage import generate_temporal_graph_filename
from sklearn.metrics import roc_auc_score
import numpy as np
from src.dgadb.data.dataset import load_df
from src.dgadb.utils import load_config

from time import time
import torch


from src.dgadb.models.TADDY.TADDY_main import TADDYModel


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

datasets = [
    "bitcoin-alpha",
    "bitcoin-otc",
    "uc-social",  # node14
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
        description="RustGraph pipeline all datasets")
    parser.add_argument("--dataset", type=str,
                        default=None, help="Dataset name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--snap_size", type=int, default=2000)

    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in datasets:
        raise RuntimeError(f"Unknown dataset: '{dataset_name}'.")

    pipeline = Pipeline.from_config(dataset_name, force_rerun=True)

    g = pipeline.run()
    temporal_graph = g.to_temporal_graph()

    temporal_graph.describe()

    anomalous_temporal_graph = inject_anomalies_addgraph_style(
        temporal_graph, anom_train_ratio=0.0, anom_val_ratio=0.0, anom_test_ratio=0.1, noise_ratio=0.0)

    evaluator = Evaluator(dataset_name=dataset_name,
                          method_name="TADDY", save_dir="eval-data")

    anomalous_temporal_graph.flip_edge_labels()

    device = "cpu"
    hyperparams = {
        "num_epoch": args.epochs,
        "snap_size": args.snap_size,
        "print_freq": 10
    }

    meta_dict = {
        "dataset_name": dataset_name,
        "train_ratio": 0.5,
        "val_ratio": 0.0,
        "anomaly_ratio": "0.5-0.0-0.05",
    }

    model = TADDYModel(device, meta_dict, hyperparams,
                       epoch_evaluation_metric=roc_auc_score)

    start_setup = time()
    model.setup_(anomalous_temporal_graph)
    time_setup = time() - start_setup

    start_train = time()
    model.train()
    time_train = time() - start_train

    preds_per_snap, labels_per_snap, *_ = model.inference(split="test")

    for snap in range(len(preds_per_snap)):
        y, pred = labels_per_snap[snap], preds_per_snap[snap]
        evaluator.eval_snapshot(torch.tensor(
            y), torch.tensor(pred), snapshot_id=snap)

    print(evaluator.get_summary())
    evaluator.save_results(setup_time=time_setup, train_time=time_train)

    total_auc_score = roc_auc_score(
        np.hstack(labels_per_snap), np.hstack(preds_per_snap))
    logger.info(f"Total test score: {total_auc_score:.4f}")

    # model.setup_(anomalous_temporal_graph)
    # model.train()

    # preds, labels, inf_time = model.inference("test")
    # auc_full = roc_auc_score(labels, preds)
    # logger.info(f"Total auc on test: {auc_full:.4f}")

    ###

# dataset = "bitcoin-alpha"

# config = load_config(dataset)
# snapshot_size = config["snapshot_size"]
# train_ratio = config["train_ratio"]
# val_ratio = config.get("val_ratio", 0.0)
# anomaly_ratio = config.get("anomaly_ratio", 0.01)

# meta_dict = {
#     "dataset_name": dataset,
#     "train_ratio": train_ratio,
#     "val_ratio": val_ratio,
#     "anomaly_ratio": anomaly_ratio,
# }

# data = load_df(dataset)
# data["edges"] = generate_data_splits(data["edges"], train_ratio, val_ratio)
# edge_features = None
# if dataset == "bitcoin-alpha" or dataset == "bitcoin-otc":
#     edge_features = data["edges"].select(["ff0_num"]).to_numpy()
#     data["edges"] = data["edges"].drop("label")

# data["edges"] = normalize_timestamps(data["edges"])

# ag = AnomalyGenerator(data["edges"], edge_features=edge_features)
# data["edges"], edge_features = ag._generate_anomalous_samples(
#     anomaly_ratio, "temporal")

# data = make_undirected(data)
# data = remove_self_loops(data)
# data = remove_duplicates(data)
# data = reindex_nodes(data)
# prev_edges = len(data["edges"])

# data = assign_snapshots(data, snapshot_size=snapshot_size)

# # print(data["edges"]) # this is exactly as after 0_prepare_data.py in TADDY

# device = "mps"
# hyperparams = {}

# model = TADDYModel(device, meta_dict, hyperparams,
#                    epoch_evaluation_metric=roc_auc_score)

# model.setup(data["edges"])
# model.train()

# preds, labels, inf_time = model.inference("test")
# auc_full = roc_auc_score(labels, preds)
# logger.info(f"Total auc on test: {auc_full:.4f}")
