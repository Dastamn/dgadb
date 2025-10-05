import os
import logging
import time
import psutil
from typing import Type

import torch

from src.dgadb.evaluation import Evaluator
from src.dgadb.storage import TemporalGraphLoader, TemporalGraphSnapshotLoader, generate_temporal_graph_filename
from src.dgadb.models.base import BaseADModel, BaseADModelComponentsType


class ExperimentRunner:
    def __init__(
            self,
            model_class: Type[BaseADModel[BaseADModelComponentsType]],
            dataset_name: str,
            model_config: dict,
            anom_config: dict,
            output_dir: str = "experiment-results",
            device: torch.device | str = "cpu"
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.anom_config = anom_config
        self.device = device

        self.model = model_class(**model_config, device=device)
        self.model_name = self.model.__class__.__name__

        self.loader = TemporalGraphLoader()
        self.data = self.loader.load(
            dataset_name, **anom_config, create_if_not_found=True)

        self.experiment_dataset_name = generate_temporal_graph_filename(
            self.data)

        self.process = psutil.Process(os.getpid())
        self.resource_logs = []

        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(
            output_dir, self.model_name, self.experiment_dataset_name, self.run_timestamp)
        os.makedirs(self.output_path, exist_ok=True)

    def _log_resources(self):
        pass

    def run(self, epochs: int, snapshot_config: dict, report_callback=None):
        self.logger.info(
            f"--- Starting Experiment: {self.model.__class__.__name__}/{self.experiment_dataset_name} ---")
        self.logger.info(f"Results will be saved to: {self.output_path}")

        train_loader = TemporalGraphSnapshotLoader(
            self.data, split="train", **snapshot_config)
        test_loader = TemporalGraphSnapshotLoader(
            self.data, split="test", **snapshot_config)
        val_loader = TemporalGraphSnapshotLoader(
            self.data, split="val", **snapshot_config)

        self.model.setup(self.data)
        self.model.train(epochs, train_loader, val_loader, report_callback)

        self.logger.info("--- Running Inference on Test Set ---")
        all_labels, all_scores = self.model.run_inference(test_loader)

        self.logger.info("--- Evaluating Results ---")
        evaluator = Evaluator(
            method_name=self.model.__class__.__name__,
            dataset_name=self.experiment_dataset_name
        )
        evaluator.eval_preds(all_labels, all_scores)
        evaluator.log_roc()
        print(evaluator.results)
        # evaluator.save_results()

        self.logger.info("--- Experiment Finished ---")


if __name__ == "__main__":
    from src.dgadb.models.baseline.GCN import GCNAD
    anom_config = {
        "anom_type": "structural",
        "anom_test_ratio": 0.1,
        "anom_val_ratio": 0.1,
    }
    snapshot_config = {
        "strategy": "window",
        "window_size": 1000,
        "include_cumulative": True
    }
    runner = ExperimentRunner(GCNAD, "bitcoin-alpha", {}, anom_config)
    runner.run(50, snapshot_config)
