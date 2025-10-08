import json
from copy import deepcopy
import os
import time
import logging
from typing import Type

import torch
from ray import tune
from ray.tune.stopper import Stopper, TrialPlateauStopper

from .utils import get_tune_run_config, get_trial_name
from .runner import ExperimentRunner
from .callbacks import ResourceMonitor, TuneReporter
from src.dgadb.models import *
from src.dgadb.storage import TemporalGraph, TemporalGraphLoader, generate_temporal_graph_filename


_BASE_PATH = os.environ["BASE_PATH"]


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
        self.output_dir = os.path.join(_BASE_PATH, output_dir)

        self.device = device
        if torch.device(self.device).type == "cuda" and self.num_gpu_per_trial == 0:
            self.logger.warning(
                "`device` is 'cuda' but `num_gpu_per_trial` is 0.")

        self.config = get_tune_run_config(
            config_path, method_name, dataset_name)
        self.model_class = self._get_model_class(self.config["model_name"])
        self.method_name = method_name

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

    def _get_model_class(self, model_name: str) -> Type[BaseADModel]:
        for sc in BaseADModel.__subclasses__():
            if model_name == sc.__name__:
                return sc
        raise ValueError(
            f"Method `{model_name}` not found, maybe check your imports?")

    def _get_param_space(self):
        hyperparams = self.config.get("hyperparameters", {})
        param_space = {}
        for key, spec in hyperparams.items():
            if isinstance(spec, dict):
                t = spec.get("search_strategy")
                space = spec.get("search_space")
                if not isinstance(space, list):
                    raise ValueError(
                        f"`search_space` should be a list for key: '{key}' and spec: '{spec}'")
                if t == "grid_search":
                    value = tune.grid_search(space)
                elif t == "random_search":
                    value = tune.choice(space)
                else:
                    raise NotImplementedError(
                        f"Unknown search strategy '{t}' for key: '{key}' and spec: '{spec}'")
            else:
                value = spec
            param_space[key] = value

        snapshot_config = self.config.get("snapshot_config", {})
        if isinstance(snapshot_config.get("window_size"), list):
            param_space["snapshot_config.window_size"] = tune.grid_search(
                snapshot_config["window_size"])

        return param_space

    def _load_checkpoint(self, checkpoint: tune.Checkpoint, model_class: Type[BaseADModel], device: torch.device | str = "cpu") -> BaseADModel:
        if checkpoint is None:
            raise ValueError(
                f"Provided checkpoint is None for model class '{model_class.__class__.__name__}'")
        with checkpoint.as_directory() as checkpoint_dir:
            self.logger.info(
                f"Loading '{self.method_name}' from checkpoint '{checkpoint_dir}'")
            return model_class.load(checkpoint_dir, self.device)

    def _save_snapshot_config(self, snapshot_config: dict, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "snapshot_config.json"), "w") as f:
            return json.dump(snapshot_config, f, indent=4)

    def _load_snapshot_config(self, load_dir: str) -> dict:
        with open(os.path.join(load_dir, "snapshot_config.json"), "r") as f:
            return json.load(f)

    def trainable_function(self, params: dict, model_class: Type[BaseADModel], data: TemporalGraph, epochs: int, snapshot_config: dict, device: torch.device | str = "cpu"):
        snapshot_config = deepcopy(snapshot_config)
        window_size = params.pop("snapshot_config.window_size", None)
        if window_size is not None:
            snapshot_config["window_size"] = window_size

        trial_dir = tune.get_context().get_trial_dir()
        self._save_snapshot_config(snapshot_config, save_dir=trial_dir)

        checkpoint = tune.get_checkpoint()
        model = (
            self._load_checkpoint(checkpoint, model_class, device)
            if checkpoint
            else model_class(**params, device=device)
        )

        runner = ExperimentRunner(
            model, data, self.dataset_name, output_dir=trial_dir)
        callbacks = [ResourceMonitor(trial_dir), TuneReporter()]
        runner.run(epochs, snapshot_config, callbacks, evaluate=False)

    def run(self, stopper: Stopper | None = None):
        param_space = self._get_param_space()

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
                    self.trainable_function,
                    model_class=self.model_class,
                    data=self.data,
                    epochs=epochs,
                    snapshot_config=snapshot_config
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
                trial_dirname_creator=get_trial_name),
            run_config=tune.RunConfig(
                name=self.experiment_name,
                storage_path=self.output_dir,
                stop=stopper,
                log_to_file=True
            )
        )

        results = tuner.fit()
        print(results)
        best_result = results.get_best_result(metric, mode, scope="last")
        self.evaluate_result(best_result)

    def evaluate_result(self, result: tune.Result):
        checkpoint = result.checkpoint
        if checkpoint is None:
            self.logger.error("No checkpoint found for evaluation.")
            return

        snapshot_config = self._load_snapshot_config(result.path)

        model = self._load_checkpoint(
            checkpoint, self.model_class, self.device)
        model.setup(self.data)

        save_dir = os.path.join(self.output_dir, self.experiment_name)
        runner = ExperimentRunner(
            model, self.data, self.dataset_name, save_dir)

        runner.evaluate(snapshot_config)


if __name__ == "__main__":
    tuner = Tuner("configs/experiments/tune.yaml", "GCN", "bitcoin-alpha")

    stopper = TrialPlateauStopper(
        tuner.metric, mode=tuner.metric_mode, grace_period=10)

    tuner.run(stopper)
