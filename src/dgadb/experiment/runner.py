import os
import logging

from .callbacks import ExperimentCallback, ResourceMonitor
from src.dgadb.evaluation import ADEvaluator
from src.dgadb.storage import TemporalGraph, TemporalGraphLoader, TemporalGraphSnapshotLoader, generate_temporal_graph_filename
from src.dgadb.models.base import BaseADModel, BaseADModelComponentsType
import torch
import numpy as np


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
            output_dir: str = "experiment-results"
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
    def run(self, epochs: int, snapshot_config: dict, callbacks: list[ExperimentCallback] | None = None, evaluate: bool = True):
        self.logger.info(
            f"Starting Experiment: {self.model.__class__.__name__}/{self.dataset_name}")
        self.logger.info(f"Results will be saved to: {self.output_dir}")

        train_loader = TemporalGraphSnapshotLoader(
            self.data, split="train", **snapshot_config)
        val_loader = TemporalGraphSnapshotLoader(
            self.data, split="val", **snapshot_config)

        self.model.setup(self.data)
        self.model.train(epochs, train_loader, val_loader, callbacks)

        if evaluate:
            self.evaluate(snapshot_config)

        self.logger.info("Done.")

    def evaluate(self, snapshot_config: dict):
        test_loader = TemporalGraphSnapshotLoader(
            self.data, split="test", **snapshot_config)
        all_labels, all_scores = self.model.run_inference(test_loader)

        evaluator = ADEvaluator(self.output_dir)
        metrics = evaluator.evaluate(all_labels, all_scores)
        self.logger.info(
            f"Evaluation result: AUC {metrics['roc_auc']}, AP {metrics['average_precision']}")
        print(
            f"Evaluation result: AUC {metrics['roc_auc']}, AP {metrics['average_precision']}")
        evaluator.save_results()


if __name__ == "__main__":
    from src.dgadb.models.baseline.gnn import GNNAD
    from src.dgadb.models.slade_new.slade import SLADEAD
    from src.dgadb.models.taddy_new.taddy import TADDYAD
    from src.dgadb.models.sad_new.sad import SADAD
    from src.dgadb.models.StrGNN.strgnn import StrGNNAD
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str,
                        default=None, help="Method name")
    parser.add_argument("--dataset", type=str,
                        default=None, help="Dataset name")

    args = parser.parse_args()

    anom_config = {
        "anom_type": "structural",
        "anom_test_ratio": 0.1,
        "anom_val_ratio": 0.1,
    }

    snapshot_config = {
        "strategy": "window",
        "window_size": 2000,
        "include_cumulative": True
    }

    if args.method == "sad":
        model = SADAD()
    elif args.method == "taddy":
        model = TADDYAD()
    elif args.method == "slade":
        model = SLADEAD()
    elif args.method == "strgnn":
        model = StrGNNAD()
    elif args.method == "gcn":
        model = GNNAD("GCN")
    elif args.method == "gat":
        model = GNNAD("GAT")
    elif args.method == "graphsage":
        model = GNNAD("GraphSAGE")
    else:
        raise NotImplementedError

    loader = TemporalGraphLoader()
    data = loader.load(
        "bitcoin-alpha", **anom_config, create_if_not_found=True)

    data = remove_duplicates(data)

    # from src.dgadb.preprocessing.pipeline.utils import load_custom_dataset
    # from src.dgadb.preprocessing.pipeline import Pipeline, StructureNormalizer

    # cont = load_custom_dataset(f"anom_gen/{args.dataset}_0.7_0.1")
    # print(f"Loaded: anom_gen/{args.dataset}_0.7_0.1")
    # pipeline = Pipeline([StructureNormalizer("canonical")])
    # data = pipeline.run(cont).to_temporal_graph()

    # data.val_mask = data.test_mask.clone()

    runner = ExperimentRunner(
        model, data, output_dir=f"latest-results/{args.method}/{args.dataset}")
    resource_monitor = ResourceMonitor(runner.output_dir, step_interval=10)
    runner.run(10, snapshot_config, [resource_monitor])
