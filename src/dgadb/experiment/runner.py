import logging
import os
from enum import Enum
from typing import Annotated

import numpy as np
import torch
import typer
from dgadb.evaluation import ADEvaluator
from dgadb.models.base import BaseADModel, BaseADModelComponentsType
from dgadb.storage import (
    TemporalGraph,
    TemporalGraphLoader,
    TemporalGraphSnapshotLoader,
    generate_temporal_graph_filename,
)

from .callbacks import ExperimentCallback, ResourceMonitor


class Method(str, Enum):
    """Available anomaly detection methods."""

    sad = "sad"
    taddy = "taddy"
    slade = "slade"
    strgnn = "strgnn"
    gcn = "gcn"
    gat = "gat"
    graphsage = "graphsage"
    rustgraph = "rustgraph"


app = typer.Typer()


def remove_duplicates(tg: TemporalGraph) -> TemporalGraph:
    """
    Removes duplicate edges from a TemporalGraph.
    This is temporary because anomaly injection can add duplicates, will be fixed soon.
    """
    if tg.num_edges == 0:
        return tg

    edge_keys = torch.stack([tg.src, tg.tgt, tg.t], dim=1).cpu().numpy()

    _, indices = np.unique(edge_keys, axis=0, return_index=True)

    indices = np.sort(indices)
    indices_torch = torch.from_numpy(indices).to(tg.device)

    tg.src = tg.src[indices_torch]
    tg.tgt = tg.tgt[indices_torch]
    tg.t = tg.t[indices_torch]
    tg.msg = tg.msg[indices_torch]
    tg.train_mask = tg.train_mask[indices_torch]
    tg.test_mask = tg.test_mask[indices_torch]

    if tg.edge_labels is not None:
        tg.edge_labels = tg.edge_labels[indices_torch]
    if tg.val_mask is not None:
        tg.val_mask = tg.val_mask[indices_torch]
    if tg.w is not None:
        tg.w = tg.w[indices_torch]

    return tg


class ExperimentRunner:
    def __init__(
        self,
        model: BaseADModel[BaseADModelComponentsType],
        data: TemporalGraph,
        dataset_name: str | None = None,
        output_dir: str = "experiment-results",
    ) -> None:
        """Orchestrates a single training and evaluation experiment.

        This class handles the end-to-end process of training a specific anomaly
        detection model on a temporal graph dataset. It manages data loading
        (splitting into train/val/test snapshots), model initialization,
        the training loop with callbacks, and final evaluation on the test set.

        Args:
            model: An instance of a model inheriting from `BaseADModel`.
            data: The `TemporalGraph` object containing the dataset.
            dataset_name: An optional name for the dataset. If None, a name is
                generated based on the graph's properties.
            output_dir: The directory where logs, plots, and evaluation results
                will be saved.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.data = data
        self.dataset_name = (
            dataset_name
            if dataset_name is not None
            else generate_temporal_graph_filename(self.data)
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # TODO: change snapshot_config to args
    def run(
        self,
        epochs: int,
        snapshot_config: dict,
        callbacks: list[ExperimentCallback] | None = None,
        evaluate: bool = True,
    ):
        self.logger.info(
            f"Starting Experiment: {self.model.__class__.__name__}/{self.dataset_name}"
        )
        self.logger.info(f"Results will be saved to: {self.output_dir}")

        train_loader = TemporalGraphSnapshotLoader(
            self.data, split="train", **snapshot_config
        )
        val_loader = TemporalGraphSnapshotLoader(
            self.data, split="val", **snapshot_config
        )

        self.model.setup(self.data)
        self.model.train(epochs, train_loader, val_loader, callbacks)

        if evaluate:
            self.evaluate(snapshot_config)

        self.logger.info("Done.")

    def evaluate(self, snapshot_config: dict):
        test_loader = TemporalGraphSnapshotLoader(
            self.data, split="test", **snapshot_config
        )
        all_labels, all_scores = self.model.run_inference(test_loader)

        evaluator = ADEvaluator(self.output_dir)
        metrics = evaluator.evaluate(all_labels, all_scores)
        self.logger.info(
            f"Evaluation result: AUC {metrics['roc_auc']}, AP {metrics['average_precision']}"
        )
        print(
            f"Evaluation result: AUC {metrics['roc_auc']}, AP {metrics['average_precision']}"
        )
        evaluator.save_results()


@app.command()
def run_experiment(
    method: Annotated[Method, typer.Option(help="Method name")],
    dataset: Annotated[str, typer.Option(help="Dataset name")] = "bitcoin-alpha",
):
    """Run an anomaly detection experiment with the specified method and dataset."""

    anom_config = {
        "anom_type": "structural",
        "anom_test_ratio": 0.1,
        "anom_val_ratio": 0.1,
    }

    snapshot_config = {
        "strategy": "window",
        "window_size": 2000,
        "include_cumulative": True,
    }

    match method:
        case Method.sad:
            from dgadb.models.sad_new.sad import SADAD

            model = SADAD()
        case Method.taddy:
            from dgadb.models.taddy_new.taddy import TADDYAD

            model = TADDYAD()
        case Method.slade:
            from dgadb.models.slade_new.slade import SLADEAD

            model = SLADEAD()
        case Method.strgnn:
            from dgadb.models.StrGNN.strgnn import StrGNNAD

            model = StrGNNAD()
        case Method.rustgraph:
            from dgadb.models.rustgraph_new.rustgraph import RustGraphAD

            model = RustGraphAD()
        case Method.gcn:
            from dgadb.models.baseline.gnn import GNNAD

            model = GNNAD("GCN")
        case Method.gat:
            from dgadb.models.baseline.gnn import GNNAD

            model = GNNAD("GAT")
        case Method.graphsage:
            from dgadb.models.baseline.gnn import GNNAD

            model = GNNAD("GraphSAGE")

    loader = TemporalGraphLoader()
    data = loader.load(dataset, **anom_config, create_if_not_found=True)
    data = remove_duplicates(data)

    runner = ExperimentRunner(
        model, data, output_dir=f"latest-results/{method.value}/{dataset}"
    )
    resource_monitor = ResourceMonitor(runner.output_dir, step_interval=10)
    runner.run(10, snapshot_config, [resource_monitor])


if __name__ == "__main__":
    app()
