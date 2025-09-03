import torch
import numpy as np
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from src.dgadb.preprocessing import Pipeline
from src.dgadb.models.RustGraph.main_RustGraph import RustGraphModel
from src.dgadb.storage import convert_temporal_graph_to_legacy_graph, Graph
from src.dgadb.preprocessing.add_graph_temporary import inject_anomalies_addgraph_style

from sklearn.metrics import roc_auc_score

from typing import Type


def tune_report(metric):
    tune.report(metrics={"metric": metric})


def training_function(config: dict, model_class: Type[RustGraphModel], graph: Graph, *args):
    meta_dict = {
        "dataset_name": "bitcoin-alpha",
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "anomaly_ratio": "0.5-0.0-0.05",
    }
    model = model_class(torch.device("cpu"), meta_dict, config, roc_auc_score)
    model.setup(graph)
    model.train(tune_report)


search_space = {
    "lr": tune.grid_search([0.001, 0.0005]),
    "epochs": 10
}


if __name__ == "__main__":
    dataset_name = "bitcoin-alpha"
    pipeline = Pipeline.from_config(dataset_name, force_rerun=True)
    container = pipeline.run()
    tg = container.to_temporal_graph()

    anomalous_temporal_graph = inject_anomalies_addgraph_style(
        tg, anom_train_ratio=0.5, anom_val_ratio=0.1, anom_test_ratio=0.0, noise_ratio=0.0)

    anomalous_temporal_graph.flip_edge_labels()

    graph = convert_temporal_graph_to_legacy_graph(anomalous_temporal_graph)
    del graph._nodes["n_feat"]
    graph.generate_snapshots(snapshot_size=2000, temporal_snapshots=False)

    tuner = tune.Tuner(
        tune.with_parameters(
            training_function, model_class=RustGraphModel, graph=graph),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=1, metric="metric", mode="max", time_budget_s=120),
    )

    results = tuner.fit()
    res = results.get_best_result("metric", "max")

    print(res)
