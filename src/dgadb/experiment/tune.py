import os
import time
import logging
from copy import deepcopy
from typing import Type

import torch
from ray import tune
from ray.tune.stopper import Stopper, TrialPlateauStopper

from . import utils
from .runner import ExperimentRunner
from .callbacks import ResourceMonitor, TuneReporter
from src.dgadb.models import BaseADModel
from src.dgadb.storage import TemporalGraph, TemporalGraphLoader, generate_temporal_graph_filename


_BASE_PATH = os.environ["BASE_PATH"]


def trainable_function(
    hyperparameters: dict,
    model_class: Type[BaseADModel],
    model_params: dict,
    data: TemporalGraph,
    epochs: int,
    snapshot_config: dict,
    dataset_name: str | None = None,
    device: torch.device | str = "cpu"
):
    snapshot_config = deepcopy(snapshot_config)
    window_size = hyperparameters.pop("snapshot_config.window_size", None)
    if window_size is not None:
        snapshot_config["window_size"] = window_size

    trial_dir = tune.get_context().get_trial_dir()
    utils.save_snapshot_config(snapshot_config, save_dir=trial_dir)

    checkpoint = tune.get_checkpoint()
    model = (
        utils.load_tune_checkpoint(
            checkpoint, model_class, model_params, device)
        if checkpoint
        else model_class(**model_params, **hyperparameters, device=device)
    )

    runner = ExperimentRunner(
        model, data, dataset_name=dataset_name, output_dir=trial_dir)
    callbacks = [ResourceMonitor(trial_dir), TuneReporter()]
    runner.run(epochs, snapshot_config, callbacks, evaluate=False)


class Tuner:
    def __init__(
            self,
            config_path: str,
            method_name: str,
            dataset_name: str,
            num_cpu_per_trial: int = 8,
            num_gpu_per_trial: int = 0,
            output_dir: str = "tune-results-new",
            loader_base_dir: str = "processed",
            device: torch.device | str = "cpu"
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.num_cpu_per_trial = num_cpu_per_trial
        self.num_gpu_per_trial = num_gpu_per_trial
        self.tune_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        self.device = device
        if torch.device(self.device).type == "cuda" and self.num_gpu_per_trial == 0:
            self.logger.warning(
                "`device` is 'cuda' but `num_gpu_per_trial` is 0.")

        self.config = utils.get_tune_run_config(
            config_path, method_name, dataset_name)
        self.model_class = utils.get_model_class(self.config["model_name"])
        self.model_params = self.config.get("model_params", {})
        self.method_name = self.config["method"]

        self.output_dir = os.path.join(
            _BASE_PATH, output_dir, method_name, dataset_name)

        self.loader = TemporalGraphLoader(loader_base_dir)
        self.data = self.loader.load(
            dataset_name, **self.config.get("anomalies", {}), create_if_not_found=True)
        self.dataset_name = generate_temporal_graph_filename(self.data)

        self.experiment_name = "_".join(
            [self.method_name, self.dataset_name, self.tune_timestamp])

    @property
    def metric(self):
        return self.config["val_metric"]["name"]

    @property
    def metric_mode(self):
        return self.config["val_metric"]["mode"]

    def run(self, stopper: Stopper | None = None):
        param_space = utils.get_tune_param_space(self.config)

        time_budget = self.config.get("time_budget", None)
        num_samples = self.config.get("num_samples", 1)
        epochs = self.config.get("epochs", None)
        snapshot_config = self.config.get("snapshot_config", None)

        metric = self.metric
        mode = self.metric_mode

        cpu_count = os.cpu_count() or 1
        gpu_count = torch.cuda.device_count()

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    trainable_function,
                    model_class=self.model_class,
                    model_params=self.model_params,
                    data=self.data,
                    epochs=epochs,
                    snapshot_config=snapshot_config,
                    dataset_name=self.dataset_name,
                    device=self.device
                ),
                resources={
                    "cpu": max(1, min(self.num_cpu_per_trial, cpu_count)),
                    "gpu": min(self.num_gpu_per_trial, gpu_count)
                }
            ),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                metric=metric,
                mode=mode,
                time_budget_s=time_budget,
                trial_dirname_creator=utils.get_tune_trial_name),
            run_config=tune.RunConfig(
                name=self.experiment_name,
                storage_path=self.output_dir,
                stop=stopper,
                # log_to_file=True
            )
        )

        results = tuner.fit()
        self.evaluate_result(results.get_best_result(metric, mode))

    def evaluate_result(self, result: tune.Result):
        checkpoint = result.checkpoint
        if checkpoint is None:
            self.logger.error("No checkpoint found for evaluation.")
            return

        snapshot_config = utils.load_snapshot_config(result.path)

        model = utils.load_tune_checkpoint(
            checkpoint, self.model_class, self.model_params, self.device)
        model.setup(self.data)

        runner = ExperimentRunner(
            model, self.data, self.dataset_name, result.path)

        runner.evaluate(snapshot_config)


if __name__ == "__main__":
    tuner = Tuner("configs/experiments/tune.yaml", "GAT", "bitcoin-alpha")

    stopper = TrialPlateauStopper(
        tuner.metric, mode=tuner.metric_mode, grace_period=10)

    tuner.run(stopper)
