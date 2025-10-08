import os
import yaml
import json
from copy import deepcopy
from typing import Type

import torch
from ray import tune
from ray.tune.experiment.trial import Trial

from src.dgadb.models import *


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


def save_snapshot_config(snapshot_config: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "snapshot_config.json"), "w") as f:
        return json.dump(snapshot_config, f, indent=4)


def load_snapshot_config(load_dir: str) -> dict:
    with open(os.path.join(load_dir, "snapshot_config.json"), "r") as f:
        return json.load(f)


def get_model_class(model_name: str) -> Type[BaseADModel]:
    for sc in BaseADModel.__subclasses__():
        if model_name == sc.__name__:
            return sc
    raise ValueError(
        f"Method `{model_name}` not found, maybe check your imports?")


def get_tune_run_config(config_path: str, method_name: str, dataset_name: str) -> dict:
    config = load_config(config_path)
    method_class_name = None
    proper_method_name = None
    for k, v in config.get("methods", {}).items():
        if k.lower() == method_name.lower():
            method_class_name = v.get("model_name", None)
            proper_method_name = k
            break

    if method_class_name is None:
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
    tune_run_config["method"] = proper_method_name

    return tune_run_config


def get_tune_trial_name(trial: Trial) -> str:
    params_str = "_".join(
        f"{k}-{v}" for k, v in trial.evaluated_params.items())
    return f"{params_str}_{trial.trial_id}"


def get_tune_param_space(config: dict):
    hyperparams = config.get("hyperparameters", {})
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

    snapshot_config = config.get("snapshot_config", {})
    if isinstance(snapshot_config.get("window_size"), list):
        param_space["snapshot_config.window_size"] = tune.grid_search(
            snapshot_config["window_size"])

    return param_space


def load_tune_checkpoint(checkpoint: tune.Checkpoint, model_class: Type[BaseADModel], model_params: dict, device: torch.device | str = "cpu") -> BaseADModel:
    with checkpoint.as_directory() as checkpoint_dir:
        return model_class.load(checkpoint_dir, device, **model_params)
