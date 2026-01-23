import json
from dgadb.evaluation import Evaluator
from dgadb.storage.temporal_graph import TemporalGraphLoader
import torch
import logging
from dgadb.preprocessing.pipeline import Pipeline
from dgadb.preprocessing.anomaly_injector import AnomalyInjector
from dgadb.storage import TemporalGraphLoader, TemporalGraphSnapshotLoader

from dgadb.storage import convert_temporal_graph_to_legacy_graph

from dgadb.data.dataset import load_df
from dgadb.models.RustGraph.main_RustGraph import RustGraphModel
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
if __name__ == "__main__":

    dataset_name = "bitcoin-alpha"

    meta_dict = {
        "dataset_name": dataset_name,
        "train_ratio": 0.5,
        "val_ratio": 0.0,
        "anomaly_ratio": "0.5-0.0-0.05",
    }
    graph_loader = TemporalGraphLoader("processed/")
    temporal_graph = graph_loader.load(
        dataset_name, create_if_not_found=True)
    print(temporal_graph)

    # temporal_graph.describe()

    graph = convert_temporal_graph_to_legacy_graph(
        temporal_graph)

    model: RustGraphModel = torch.load("test.pt")
    model.setup(graph)

    # model = RustGraphModel("cpu", meta_dict, {}, None)
    # torch.save(model, "test.pt")

    # config_name = "yelp-zip-example"

    # dataset_name = "bitcoin-alpha"

    # anom_type = "temporal"

    # gen_params = {"distance_metric": "l2"}
    # gen_params = {}

    # print(graph.n_feat)

    # temporal_graph = g.to_temporal_graph()
    # temporal_graph.describe()

    # anom_injector = AnomalyInjector(temporal_graph)

    # anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
    #     "c", anom_train_ratio=0.05, anom_test_ratio=0.05, anom_val_ratio=0.05
    # )

    # snapshot_loader = TemporalGraphSnapshotLoader(
    #     anomalous_temporal_graph, strategy="window", window_size=1000)

    # print("Number of snapshots: ", len(snapshot_loader))

    # for snap in snapshot_loader:
    #     print(snap)
