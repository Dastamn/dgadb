import logging
from src.dgadb.preprocessing.pipeline import Pipeline
from src.dgadb.preprocessing.anomaly_injector import AnomalyInjector
from src.dgadb.data.builder import build_graph_from_temporal
from src.dgadb.models.SAD.main_SAD import SADModel
from src.dgadb.evaluation.evaluator import Evaluator
from sklearn.metrics import roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


if __name__ == "__main__":
    config_name = "mooc"

    pipeline = Pipeline.from_config(config_name, force_rerun=False)
    evaluator = Evaluator(dataset_name="mooc", method_name="RustGraph", output_dir="eval-data")

    g = pipeline.run()
    g.describe()

    temporal_graph = g.to_temporal_graph()
    temporal_graph.describe()

    # TODO @Dastamn: Test on GPU
    anom_injector = AnomalyInjector(temporal_graph)

    anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
        "s", anom_train_ratio=0.05, anom_test_ratio=0.05, anom_val_ratio=0.05, reset_labels=False
    )

    graph = build_graph_from_temporal(anomalous_temporal_graph)
    # use node2vec embs
    graph._nodes["n_feat"] = torch.zeros((graph.num_nodes, 4), dtype=torch.float32)
    meta = anomalous_temporal_graph.metadata

    meta_dict = {
        "dataset_name": meta["dataset_name"],
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "anomaly_ratio": 0.05,
    }
    # TODO: meta_dict should be produced from the pipeline

    device = "cpu"
    hyperparams = {"num_epochs": 10, "input_dim": 4, "num_data_workers": 4}
    model = SADModel(device, meta_dict, hyperparams, roc_auc_score)
    model.setup(graph)
    model.train()
    preds, labels = model.inference(split="test")
    total_auc_score = roc_auc_score(labels, preds)
    logger.info(f"Total test score: {total_auc_score:.4f}")
