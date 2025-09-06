from pathlib import Path
import json
import logging
from src.dgadb.preprocessing.pipeline.pipeline import Pipeline
from src.dgadb.models.RustGraph.main_RustGraph import RustGraphModel
from src.dgadb.storage import convert_temporal_graph_to_legacy_graph, Graph
from src.dgadb.preprocessing.add_graph_temporary import inject_anomalies_addgraph_style
import os
import yaml
from sklearn.metrics import roc_auc_score
import torch
from ray import train, tune
from typing import Type
import multiprocessing

_BASE_PATH = os.environ["BASE_PATH"]
_CONFIG_PATH = os.path.join(_BASE_PATH, "configs")

_MODELS = {
    "RustGraph": RustGraphModel
}


class Experiment2:
    def __init__(self, config_name: str, config_dir: str = "configs/experiments") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)


class Experiment():
    def __init__(self, exp_name: str, dataset_name: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exp_config = self._load_config(exp_name)
        self.model = _MODELS[self.exp_config["model"]]
        self.num_samples = self.exp_config.get("num_samples", 1)
        # self.time_budget_s = self.exp_config.get("time_budget_s", 3600)
        self.time_budget_s = self.exp_config.get("time_budget_s", 60)
        self.dataset_name = dataset_name
        self.param_space = self._load_param_space(self.exp_config)

    def _load_config(self, config_name):
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        config_path = os.path.join(_CONFIG_PATH, "experiments", config_name)
        self.logger.info(f"Loading config: '{config_path}'")

        if not os.path.exists(config_path):
            error = FileNotFoundError(
                f"Config file not found: '{config_path}'")
            self.logger.error(error)
            raise error

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        except yaml.YAMLError as e:
            self.logger.error(
                f"Error parsing YAML config '{config_path}': {e}")
            raise

    def _load_param_space(self, exp_config):

        hp = exp_config.get("hyperparameters", {})
        param_space = {}

        for key, spec in hp.items():

            if isinstance(spec, dict):
                if self.dataset_name in spec:
                    block = spec[self.dataset_name]
                elif "default" in spec:
                    block = spec["default"]
                else:
                    block = spec
            else:
                block = spec

            value = None
            if isinstance(block, dict):
                if "fixed" in block:
                    value = block["fixed"]
                else:
                    t = block.get("type")
                    space = block.get("search_space")
                    if t == "grid_search":
                        value = tune.grid_search(space)
                    elif t == "random_search":
                        value = tune.choice(space)
                    else:
                        value = block
            else:
                value = block

            param_space[key] = value

        return param_space

    def preprocessing(self, flip_labels: bool):
        pipeline, meta_dict = Pipeline.from_config(
            self.dataset_name, force_rerun=True)
        self.meta_dict = meta_dict
        self.meta_dict["dataset_name"] = self.dataset_name
        container = pipeline.run()
        tg = container.to_temporal_graph()

        anomalous_temporal_graph = inject_anomalies_addgraph_style(
            tg,
            anom_train_ratio=self.meta_dict.get("anom_train_ratio", 0.0),
            anom_val_ratio=self.meta_dict.get("anom_val_ratio", 0.0),
            anom_test_ratio=self.meta_dict.get("anom_test_ratio", 0.0),
            noise_ratio=0.0)

        if flip_labels:
            anomalous_temporal_graph.flip_edge_labels()

        self.graph = convert_temporal_graph_to_legacy_graph(
            anomalous_temporal_graph)
        del self.graph._nodes["n_feat"]

    def training_function(self, config: dict, model_class: Type[RustGraphModel], graph: Graph, meta_dict: dict, *args):
        def tune_report(metric):
            tune.report(metrics={"metric": metric})

        graph.generate_snapshots(
            snapshot_size=config["snapshot_size"], temporal_snapshots=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_class(device, meta_dict,
                            config, roc_auc_score)
        graph = graph.to(device)
        model.setup(graph)
        model.train(tune_report)

    def run(self):
        cpu_count = multiprocessing.cpu_count()
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    self.training_function, model_class=RustGraphModel, graph=self.graph, meta_dict=self.meta_dict),
                {"cpu": cpu_count // 2}),
            param_space=self.param_space,
            tune_config=tune.TuneConfig(
                num_samples=self.num_samples, metric="metric", mode="max", time_budget_s=self.time_budget_s),
        )

        results = tuner.fit()
        print(results)
        best_result = results.get_best_result("metric", "max")

        print(best_result)

        output = {
            "model": self.exp_config["model"],
            "dataset": self.dataset_name,
            "best_config": best_result.config,
            # "log_path": best_result.log_dir,
            # "total_time_s": best_result.time_total_s,
        }

        if best_result.metrics is not None and "metric" in best_result.metrics:
            output["metric"] = best_result.metrics["metric"]

        # Save results
        output_dir = Path("results") / self.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.exp_config['model']}_results.json"

        self.logger.info(f"Saving best result to {output_file}")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)

        print("--- Best Result ---")
        print(json.dumps(output, indent=4))


if __name__ == "__main__":
    dataset_name = "reddit"
    e = Experiment(exp_name="rustgraph_test", dataset_name=dataset_name)
    e.preprocessing(flip_labels=True)
    e.run()
