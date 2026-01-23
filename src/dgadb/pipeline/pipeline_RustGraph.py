import logging
from dgadb.preprocessing.pipeline import Pipeline
from dgadb.preprocessing.anomaly_injector import AnomalyInjector
from dgadb.preprocessing.add_graph_temporary import inject_anomalies_addgraph_style
from dgadb.data.builder import build_graph_from_temporal
from dgadb.models.RustGraph.main_RustGraph import RustGraphModel
from dgadb.evaluation.evaluator import Evaluator
from sklearn.metrics import roc_auc_score
import numpy as np
logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


if __name__ == "__main__":
    # config_name = "yelp-zip-example"
    config_name = "bitcoin-alpha-example"

    pipeline = Pipeline.from_config(config_name, force_rerun=False)
    evaluator = Evaluator(dataset_name="bitcoin-alpha",
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
    model.setup(graph)
    model.train()
    preds_per_snap, labels_per_snap = model.inference(split="test")
    for snap in range(len(preds_per_snap)):
        y, pred = labels_per_snap[snap], preds_per_snap[snap]
        evaluator.eval_snapshot(y, pred, snapshot_id=snap)
    print(evaluator.get_summary())

    total_auc_score = roc_auc_score(
        np.hstack(labels_per_snap), np.hstack(preds_per_snap))
    logger.info(f"Total test score: {total_auc_score:.4f}")
