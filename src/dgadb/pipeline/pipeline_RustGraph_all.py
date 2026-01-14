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
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

datasets = [
    "bitcoin-alpha",  # node1
    "bitcoin-otc",  # node1
    "uc-social",  # node2
    "digg-homo",  # node3
    "as-topology",  # node3
    "email-dnc",  # node4
    "enron",  # node5
    "epinions",  # node6
    "mooc",  # node1
    "reddit",  # node2
    "dgraph",  # node9
    "wiki"  # node3
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RustGraph pipeline all datasets")
    parser.add_argument("--dataset", type=str,
                        default=None, help="Dataset name")
    parser.add_argument("--epochs", type=int, default=250)

    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in datasets:
        raise RuntimeError(f"Unknown dataset: '{dataset_name}'.")

    # config_name = "yelp-zip-example"
    config_name = dataset_name

    pipeline = Pipeline.from_config(config_name, force_rerun=False)
    evaluator = Evaluator(dataset_name=dataset_name,
                          method_name="RustGraph", save_dir="eval-data")

    g = pipeline.run()
    g.describe()

    temporal_graph = g.to_temporal_graph()
    temporal_graph.describe()

    # TODO @Dastamn: Test on GPU
    anom_injector = AnomalyInjector(temporal_graph)

    # anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
    #     "s", anom_train_ratio=0.5, anom_test_ratio=0.05, anom_val_ratio=0.0, reset_labels=True)

    anomalous_temporal_graph = inject_anomalies_addgraph_style(
        temporal_graph, anom_train_ratio=0.5, anom_val_ratio=0.0, anom_test_ratio=0.05, noise_ratio=0.0)

    # anomalous_temporal_graph.flip_edge_labels()

    graph = build_graph_from_temporal(anomalous_temporal_graph)
    # use node2vec embs
    del graph._nodes["n_feat"]
    graph.generate_snapshots(snapshot_size=2000, temporal_snapshots=False)
    meta = anomalous_temporal_graph.metadata

    meta_dict = {
        "dataset_name": meta["dataset_name"],
        "train_ratio": 0.5,
        "val_ratio": 0.0,
        "anomaly_ratio": "0.5-0.0-0.05",
    }
    # TODO: meta_dict should be produced from the pipeline

    device = "cpu"
    hyperparams = {
        "num_epochs": 250,
        "snap_size": 2000,
        "x_dim": 128,
        "h_dim": 128,
        "z_dim": 128,
        "print_freq": 10
    }
    model = RustGraphModel(device, meta_dict, hyperparams, roc_auc_score)

    start_setup = time()
    model.setup(graph)
    time_setup = time() - start_setup

    start_train = time()
    model.train()
    time_train = time() - start_train

    preds_per_snap, labels_per_snap = model.inference(split="test")

    for snap in range(len(preds_per_snap)):
        y, pred = labels_per_snap[snap], preds_per_snap[snap]
        evaluator.eval_snapshot(y, pred, snapshot_id=snap)

    print(evaluator.get_summary())
    evaluator.save_results(setup_time=time_setup, train_time=time_train)

    total_auc_score = roc_auc_score(
        np.hstack(labels_per_snap), np.hstack(preds_per_snap))
    logger.info(f"Total test score: {total_auc_score:.4f}")


###


# if __name__ == "__main__":

#     datasets = [
#         "bitcoin-alpha",
#         "bitcoin-otc",
#         "uc-social",
#         "digg-homo",
#         "as-topology",
#         "email-dnc",
#         "enron",
#         "epinions",
#         "mooc",
#         "reddit",
#         "dgraph",
#         "wiki"
#     ]

#     for dataset_name in datasets:

#         # config_name = "yelp-zip-example"
#         # config_name = "bitcoin-alpha-example"

#         pipeline = Pipeline.from_config(dataset_name, force_rerun=False)

#         g = pipeline.run()
#         g.describe()

#         temporal_graph = g.to_temporal_graph()
#         temporal_graph.describe()

#         # TODO @Dastamn: Test on GPU
#         anom_injector = AnomalyInjector(temporal_graph)

#         anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
#             "s", anom_train_ratio=0.05, anom_test_ratio=0.05, anom_val_ratio=0.05, reset_labels=True)

#         anom_dataset_filename = generate_temporal_graph_filename(
#             temporal_graph)

#         evaluator = Evaluator(dataset_name=anom_dataset_filename,
#                               method_name="RustGraph", output_dir="eval-data")

#         anomalous_temporal_graph.flip_edge_labels()

#         graph = build_graph_from_temporal(anomalous_temporal_graph)
#         # use node2vec embs
#         del graph._nodes["n_feat"]
#         graph.generate_snapshots(snapshot_size=1000, temporal_snapshots=False)
#         meta = anomalous_temporal_graph.metadata

#         meta_dict = {
#             "dataset_name": meta["dataset_name"],
#             "train_ratio": 0.7,
#             "val_ratio": 0.1,
#             "anomaly_ratio": 0.05,
#         }
#         # TODO: meta_dict should be produced from the pipeline

#         device = anomalous_temporal_graph.device
#         hyperparams = {
#             "num_epochs": 1,
#             "x_dim": 128,
#             "h_dim": 128,
#             "z_dim": 128
#         }
#         model = RustGraphModel(device, meta_dict, hyperparams, roc_auc_score)

#         start_setup = time()
#         model.setup(graph)
#         time_setup = time() - start_setup

#         start_train = time()
#         model.train()
#         time_train = time() - start_train

#         preds_per_snap, labels_per_snap = model.inference(split="test")
#         for snap in range(len(preds_per_snap)):
#             y, pred = labels_per_snap[snap], preds_per_snap[snap]
#             evaluator.eval_snapshot(y, pred, snapshot_id=snap)

#         print(evaluator.get_summary())
#         evaluator.save_results(setup_time=time_setup, train_time=time_train)

#         total_auc_score = roc_auc_score(
#             np.hstack(labels_per_snap), np.hstack(preds_per_snap))
#         logger.info(f"Total test score: {total_auc_score:.4f}")
