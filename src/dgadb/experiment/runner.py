import os
import logging

from .callbacks import ExperimentCallback, ResourceMonitor
from src.dgadb.evaluation import Evaluator
from src.dgadb.storage import TemporalGraph, TemporalGraphLoader, TemporalGraphSnapshotLoader, generate_temporal_graph_filename
from src.dgadb.models.base import BaseADModel, BaseADModelComponentsType


class ExperimentRunner:
    def __init__(
            self,
            model: BaseADModel[BaseADModelComponentsType],
            data: TemporalGraph,
            dataset_name: str | None = None,
            output_dir: str = "experiment-results"
    ) -> None:
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

    def run(self, epochs: int, snapshot_config: dict, callbacks: list[ExperimentCallback] | None = None, evaluate: bool = True):
        self.logger.info(
            f"--- Starting Experiment: {self.model.__class__.__name__}/{self.dataset_name} ---")
        self.logger.info(f"Results will be saved to: {self.output_dir}")

        train_loader = TemporalGraphSnapshotLoader(
            self.data, split="train", **snapshot_config)
        val_loader = TemporalGraphSnapshotLoader(
            self.data, split="val", **snapshot_config)

        self.model.setup(self.data)
        self.model.train(epochs, train_loader, val_loader, callbacks)

        if evaluate:
            self.evaluate()

        self.logger.info("--- Experiment Finished ---")

    def evaluate(self):
        self.logger.info("--- Running Inference on Test Set ---")
        test_loader = TemporalGraphSnapshotLoader(
            self.data, split="test", **snapshot_config)
        all_labels, all_scores = self.model.run_inference(test_loader)

        self.logger.info("Evaluating Results...")
        evaluator = Evaluator(
            method_name=self.model.__class__.__name__,
            dataset_name=self.dataset_name
        )
        evaluator.eval_preds(all_labels, all_scores)
        evaluator.log_roc()
        evaluator.save_results()
        print(evaluator.results)


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

    model = GCNAD()

    loader = TemporalGraphLoader()
    data = loader.load(
        "bitcoin-alpha", **anom_config, create_if_not_found=True)

    runner = ExperimentRunner(model, data)
    resource_monitor = ResourceMonitor(runner.output_dir, step_interval=10)
    runner.run(50, snapshot_config, [resource_monitor])
