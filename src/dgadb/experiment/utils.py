from ray.tune.experiment.trial import Trial
import os
import yaml
from copy import deepcopy

from src.dgadb.models.baseline.GCN import GCNAD


def deep_update(input_dict: dict, update_dict: dict):
    for k, v in update_dict.items():
        if isinstance(v, dict):
            input_dict[k] = deep_update(input_dict.get(k, {}), v)
        else:
            input_dict[k] = v
    return input_dict


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: '{config_path}'")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(
            f"Error parsing YAML config '{config_path}': {e}")


def get_tune_run_config(config_path: str, method_name: str, dataset_name: str) -> dict:
    config = load_config(config_path)
    methos_class_name = None
    proper_method_name = None
    for k, v in config.get("methods", {}).items():
        if k.lower() == method_name.lower():
            methos_class_name = v.get("model_name", None)
            proper_method_name = k
            break

    if methos_class_name is None:
        raise ValueError(
            f"Method '{method_name}' not found in config file.")

    tune_run_config = deepcopy(config.get('global', {}))

    dataset_config = config.get('datasets', {}).get(dataset_name, {})
    tune_run_config = deep_update(tune_run_config, dataset_config)

    method_config = config['methods'][proper_method_name]
    tune_run_config = deep_update(tune_run_config, method_config)

    override_config = method_config.get(
        'dataset_overrides', {}).get(dataset_name, {})
    tune_run_config = deep_update(tune_run_config, override_config)

    tune_run_config.pop('overrides', None)

    return tune_run_config


def get_trial_name(trial: Trial) -> str:
    params_str = "_".join(
        f"{k}-{v}" for k, v in trial.evaluated_params.items())
    return f"{params_str}_{trial.trial_id}"


if __name__ == "__main__":
    from src.dgadb.models.base import BaseADModel
    from src.dgadb.models.baseline.GCN import GCNAD
    subclasses = BaseADModel.__subclasses__()
    for s in subclasses:
        print(s.__name__)
