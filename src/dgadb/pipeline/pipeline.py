import json
from src.dgadb.evaluation import Evaluator
from src.dgadb.storage.temporal_graph import TemporalGraphLoader
import torch
import logging
from src.dgadb.preprocessing.pipeline import Pipeline
from src.dgadb.preprocessing.anomaly_injector import AnomalyInjector
from src.dgadb.storage import TemporalGraphLoader, TemporalGraphSnapshotLoader

from src.dgadb.storage import convert_temporal_graph_to_legacy_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
if __name__ == "__main__":

    dataset_name = "yelp-zip"
    # config_name = "yelp-zip-example"

    # dataset_name = "bitcoin-alpha"

    anom_type = "temporal"

    # gen_params = {"distance_metric": "l2"}
    # gen_params = {}

    # graph_loader = TemporalGraphLoader("processed/")
    # temporal_graph = graph_loader.load(
    #     dataset_name, create_if_not_found=True, device="mps", **gen_params)

    # temporal_graph.describe()

    # graph = convert_temporal_graph_to_legacy_graph(
    #     temporal_graph)
    # print(graph.n_feat)

    pipeline = Pipeline.from_config(dataset_name, force_rerun=False)

    g = pipeline.run()
    g.describe()

    temporal_graph = g.to_temporal_graph()
    temporal_graph.describe()

    anom_injector = AnomalyInjector(temporal_graph)

    anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
        "c", anom_train_ratio=0.05, anom_test_ratio=0.05, anom_val_ratio=0.05
    )

    snapshot_loader = TemporalGraphSnapshotLoader(
        anomalous_temporal_graph, strategy="window", window_size=1000)

    # print("Number of snapshots: ", len(snapshot_loader))

    # for snap in snapshot_loader:
    #     print(snap)
