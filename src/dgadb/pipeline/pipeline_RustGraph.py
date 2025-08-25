import logging
from src.dgadb.preprocessing.pipeline import Pipeline
from src.dgadb.preprocessing.anomaly_injector import AnomalyInjector
from src.dgadb.data.builder import build_graph_from_temporal
from src.dgadb.models.RustGraph.main_RustGraph import RustGraphModel
from sklearn.metrics import roc_auc_score
logger = logging.getLogger(__name__)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


if __name__ == "__main__":
    # config_name = "yelp-zip-example"
    config_name = "bitcoin-alpha-example"

    pipeline = Pipeline.from_config(config_name, force_rerun=False)

    g = pipeline.run()
    g.describe()

    temporal_graph = g.to_temporal_graph()
    temporal_graph.describe()

    # TODO @Dastamn: Test on GPU
    anom_injector = AnomalyInjector(temporal_graph)

    anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
        "c", anom_train_ratio=0.05, anom_test_ratio=0.05, anom_val_ratio=0.05)

    print(anomalous_temporal_graph)

    graph = build_graph_from_temporal(anomalous_temporal_graph)
    graph.generate_snapshots(snapshot_size=1000, temporal_snapshots=False)

    meta_dict = {
        "dataset_name":"bitcoin_alpha",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "anomaly_ratio": 0.05,
    }
    # TODO: meta_dict should be produced from the pipeline

    device = "cpu"
    hyperparams = {"num_epochs":3}
    model = RustGraphModel(device, meta_dict, hyperparams, roc_auc_score)
    model.setup(graph)
    model.train()
    preds, labels, inf_time = model.inference(split="test")
    auc_score = roc_auc_score(labels, preds)
    logger.info(f"Test Score: {auc_score:.4f}, took {inf_time:.4f} seconds")

