

import json
from datetime import datetime
from dgadb.models.SAD.main_SAD import SADModel
from dgadb.evaluation import Evaluator, evaluate, VALID_METRICS
import tempfile
import torch
import multiprocessing
from typing import Optional, Literal
from dgadb.storage import TemporalGraph, TemporalGraphLoader, generate_temporal_graph_filename
from dgadb.preprocessing import Pipeline
import argparse
from ray import tune
import logging
from .config_loader import load_config, get_run_config
import os
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, Stopper, TrialPlateauStopper, ExperimentPlateauStopper
from sklearn.metrics import roc_auc_score
from ray.tune import Checkpoint
from dgadb.preprocessing.add_graph_temporary import inject_anomalies_addgraph_style

from dgadb.models.TADDY.TADDY_main import TADDYModel
from dgadb.models.StrGNN.StrGNN_main import STRGNNModel
from dgadb.models.GeneralDYG.GeneralDYG_main import GeneralDYGModel

from dgadb.models.baseline.Node2Vec import n2vModel
from dgadb.models.baseline.NetwalkBaseline import NetWalkBaseline
from dgadb.models.baseline.GNNBaseline import GNNBaseline
from dgadb.models.SAD.main_SAD import SADModel
from dgadb.models.RustGraph.main_RustGraph import RustGraphModel


from dgadb.utils.paths import get_project_root

_BASE_PATH = get_project_root()

_MODELS = {
    "rustgraph": RustGraphModel,
    "taddy": TADDYModel,
    "strgnn": STRGNNModel,
    "sad": SADModel,
    "generaldyg": GeneralDYGModel,
    "node2vec": n2vModel,
    "netwalk": NetWalkBaseline,
    # "GNNBaseline": GNNBaseline,
}


class Tuner:
    def __init__(
        self,
        config_path: str,
        method_name: str,
        dataset_name: str,
        metric_name: str,
        mode: Literal["min", "max"] = "max",
        results_dir: str = "tune-results/",
        # checkpoint_dir: str = "model-checkpoint/",
        *args
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if metric_name not in VALID_METRICS:
            e = ValueError(
                f"Invalid 'metric_name', got: {method_name}, expected {VALID_METRICS}.")
            self.logger.error(e)
            raise e

        # self.results_dir = os.path.join(_BASE_PATH, results_dir)
        # self.checkpoint_dir = os.path.join(_BASE_PATH, checkpoint_dir)
        start_time = datetime.now()
        self.exp_dir = os.path.join(
            _BASE_PATH,
            results_dir,
            method_name,
            dataset_name,
            start_time.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.exp_dir, exist_ok=True)
        self.logger.info(f"Tuner results will be saved to: '{self.exp_dir}'")

        self.metric_name = metric_name
        self.mode: Literal["min", "max"] = mode

        self.config_path = config_path
        self.config = load_config(config_path)
        self.run_config = get_run_config(
            self.config, method_name, dataset_name)

        self.method_name = method_name
        # self.dataset_name = dataset_name

        tg_loader = TemporalGraphLoader(*args)
        anom_conf = self.run_config["anomalies"]

        self.tg = tg_loader.load(
            dataset_name,
            # anom_type=anom_conf["anom_type"],
            # anom_train_ratio=anom_conf["anom_train_ratio"],
            # anom_val_ratio=anom_conf["anom_val_ratio"],
            # anom_test_ratio=anom_conf["anom_test_ratio"],
            create_if_not_found=True,
            # reset_labels=True
        )
        if any([anom_conf["anom_train_ratio"], anom_conf["anom_val_ratio"], anom_conf["anom_test_ratio"]]):
            self.logger.info(f"Injecting anomalies: {anom_conf}")
            self.tg = inject_anomalies_addgraph_style(
                self.tg,
                anom_train_ratio=anom_conf["anom_train_ratio"],
                anom_val_ratio=anom_conf["anom_val_ratio"],
                anom_test_ratio=anom_conf["anom_test_ratio"],
            )
        # tg_loader.save(self.tg)
        # self.dataset_name = generate_temporal_graph_filename(self.tg)
        self.dataset_name = dataset_name

        if anom_conf["anomaly_as_0"]:
            self.tg.flip_edge_labels()

    def _get_ray_search_space(self):
        hyperparams = self.run_config.get("hyperparameters", {})
        param_space = {}

        for key, spec in hyperparams.items():
            value = None
            if isinstance(spec, dict):
                if "fixed" in spec:
                    value = spec["fixed"]
                else:
                    t = spec.get("type")
                    space = spec.get("search_space")
                    if not isinstance(space, list):
                        raise ValueError(
                            f"'search_space' should be a list for key: {key} and spec: {spec}.")

                    if t == "grid_search":
                        value = tune.grid_search(space)
                    elif t == "random_search":
                        value = tune.choice(space)
                    else:
                        value = spec
            else:
                value = spec

            param_space[key] = value

        return param_space

    def run_model(self, config: dict, tg: TemporalGraph):
        best_val_metric = -float('inf') if self.mode == "max" else float('inf')

        def report(metric_value, model, epoch, save=True):
            assert model is not None
            nonlocal best_val_metric
            best_val_metric = (
                max(best_val_metric, metric_value)
                if self.mode == "max"
                else min(best_val_metric, metric_value)
            )

            checkpoint = None
            if save:
                checkpoint_dir = os.path.join(
                    "checkpoint",
                    tune.get_context().get_trial_id(),
                    f"epoch_{epoch}"
                )
                # checkpoint_dir = os.path.join(
                #     self.checkpoint_dir,
                #     self.method_name,
                #     self.dataset_name,
                #     tune.get_context().get_trial_id(),
                #     f"epoch_{epoch}"
                # )
                os.makedirs(checkpoint_dir, exist_ok=True)

                # try:
                torch.save(
                    model, os.path.join(checkpoint_dir, "model.pt"))
                checkpoint = Checkpoint.from_directory(checkpoint_dir)

                # except TypeError as e:
                #     self.logger.error(f"Error when saving model: {e}")

            preds, labels = model.inference("test")
            metrics = evaluate(
                labels,
                preds,
                self.run_config.get("anomalies", {}).get("anomaly_as_0", False)
            )
            tune.report(
                metrics={
                    f"val_{self.metric_name}": best_val_metric,
                    **{f"test_{k}": v for k, v in metrics.items()}
                },
                checkpoint=checkpoint
            )

            # tune.report(
            #     metrics={f"val_{self.metric_name}": metric_value},
            #     checkpoint=checkpoint
            # )

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        meta_dict = {
            "dataset_name": self.dataset_name,
            **self.tg.metadata.get("splits", {}),
            **self.run_config.get("anomalies", {})
        }
        model = _MODELS[self.method_name](
            device, meta_dict, config, roc_auc_score)
        model.setup(tg)
        model.train(report)
        # preds, labels = model.inference("test")
        # metrics = evaluate(
        #     labels,
        #     preds,
        #     self.run_config.get("anomalies", {}).get("anomaly_as_0", False)
        # )
        # tune.report(
        #     metrics={
        #         f"val_{self.metric_name}": best_val_metric,
        #         **{f"test_{k}": v for k, v in metrics.items()}
        #     }
        # )

    def run_tuner(self, stopper: Optional[Stopper] = None):
        num_trials = self.run_config.get(
            "search_strategy", {}).get("num_trials", 1)
        time_budget = self.run_config.get("time_budget", None)
        param_space = self._get_ray_search_space()
        cpu_count = multiprocessing.cpu_count()

        # with tempfile.TemporaryDirectory() as tmpdir:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    self.run_model, tg=self.tg),
                {"cpu": max(1, cpu_count // num_trials),
                    "gpu": min(1, torch.cuda.device_count())}
            ),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=num_trials, metric=f"val_{self.metric_name}", mode=self.mode, time_budget_s=time_budget),
            run_config=tune.RunConfig(
                stop=stopper,
                name=f"tune_{self.method_name}_{self.dataset_name}",
                # storage_path=tmpdir,
                storage_path=self.exp_dir,
                log_to_file=True
            )
        )

        tune_results = tuner.fit()
        best_result = tune_results.get_best_result(
            f"val_{self.metric_name}", self.mode)

        best_metrics = best_result.metrics
        if best_metrics is None or not best_metrics.get("done", False):
            self.logger.error(
                "Tuning finished, but no best result was found. Check trial logs for errors.")
            return

        best_trial_summary_dir = os.path.join(
            self.exp_dir, "best_trial_summary")
        os.makedirs(best_trial_summary_dir, exist_ok=True)
        self.logger.info(
            f"Saving best trial summary to: {best_trial_summary_dir}")

        # target_dir = os.path.join(
        #     self.results_dir,
        #     self.method_name,
        #     self.dataset_name,
        #     start_datetime.strftime("%Y%m%d_%H%M%S/")
        # )
        # os.makedirs(target_dir, exist_ok=True)

        self.logger.info("Saving best metrics...")
        with open(os.path.join(best_trial_summary_dir, "metrics.json"), "w") as f:
            json.dump(best_metrics, f, indent=4)

        best_checkpoint = best_result.checkpoint
        if best_checkpoint is not None:
            self.logger.info("Saving best checkpoint...")
            with best_checkpoint.as_directory() as checkpoint_dir:
                # Copy model
                best_model_path = os.path.join(checkpoint_dir, "model.pt")
                if os.path.exists(best_model_path):
                    import shutil
                    shutil.copy(best_model_path, os.path.join(
                        best_trial_summary_dir, "model.pt"))
                else:
                    self.logger.warning(
                        f"model.pt not found in checkpoint dir: {checkpoint_dir}")
                # best_model = torch.load(
                #     os.path.join(checkpoint_dir, "model.pt"))
                # torch.save(best_model, os.path.join(
                #     best_trial_summary_dir, "model.pt"))
        else:
            self.logger.warning("No checkpoint to save.")

        print(best_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuner")
    parser.add_argument("--method", type=str,
                        default=None, help="Method name")
    parser.add_argument("--dataset", type=str,
                        default=None, help="Dataset name")

    args = parser.parse_args()

    tuner = Tuner("configs/tune/tune.yaml", args.method,
                  args.dataset, "roc_auc")

    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=5)
    )

    stopper = CombinedStopper(
        TrialPlateauStopper(
            std=0.001,
            metric="metric",
            grace_period=10
        ),
        MaximumIterationStopper(20)
    )

    tuner.run_tuner(stopper)
