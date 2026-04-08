import os
import yaml
import json
from copy import deepcopy
from typing import Type

import torch
from ray import tune
from ray.tune.experiment.trial import Trial

from dgadb.models import *


def deep_update(input_dict: dict, update_dict: dict):
    """Recursively merge ``update_dict`` into ``input_dict``.

    Nested dicts are merged rather than replaced. Returns ``input_dict``
    after in-place modification.

    Args:
        input_dict: Base dictionary to update.
        update_dict: Values to merge in; nested dicts are merged recursively.

    Returns:
        The updated ``input_dict``.
    """
    for k, v in update_dict.items():
        if isinstance(v, dict):
            input_dict[k] = deep_update(input_dict.get(k, {}), v)
        else:
            input_dict[k] = v
    return input_dict


def load_config(config_path: str) -> dict:
    """Load and parse a YAML config file.

    Args:
        config_path: Absolute or relative path to the YAML file.

    Returns:
        Parsed config as a plain dict.

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If the file cannot be parsed as valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: '{config_path}'")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(
            f"Error parsing YAML config '{config_path}': {e}")


def save_snapshot_config(snapshot_config: dict, save_dir: str) -> None:
    """Persist a snapshot config dict to ``<save_dir>/snapshot_config.json``.

    Args:
        snapshot_config: Dict of keyword arguments for
            :class:`TemporalGraphSnapshotLoader`.
        save_dir: Directory to write into (created if absent).
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "snapshot_config.json"), "w") as f:
        return json.dump(snapshot_config, f, indent=4)


def load_snapshot_config(load_dir: str) -> dict:
    """Load a snapshot config dict from ``<load_dir>/snapshot_config.json``.

    Args:
        load_dir: Directory containing ``snapshot_config.json``.

    Returns:
        The snapshot config dict.
    """
    with open(os.path.join(load_dir, "snapshot_config.json"), "r") as f:
        return json.load(f)


def get_model_class(model_name: str) -> Type[BaseADModel]:
    """Return the :class:`BaseADModel` subclass whose name matches ``model_name``.

    Args:
        model_name: Exact class name string (case-sensitive).

    Returns:
        The matching :class:`BaseADModel` subclass.

    Raises:
        ValueError: If no registered subclass matches.
    """
    for sc in BaseADModel.__subclasses__():
        if model_name == sc.__name__:
            return sc
    raise ValueError(
        f"Method `{model_name}` not found, maybe check your imports?")


def get_tune_run_config(config_path: str, method_name: str, dataset_name: str) -> dict:
    """Build a merged tune-run config for a specific method and dataset.

    Reads the YAML at ``config_path``, then applies (in order): global
    defaults, dataset-level overrides, method-level overrides, and
    dataset-specific method overrides.

    Args:
        config_path: Path to the top-level tune config YAML.
        method_name: Case-insensitive method key (e.g. ``"gcn"``).
        dataset_name: Dataset key as used in the config file.

    Returns:
        A flat dict ready to pass to Ray Tune.

    Raises:
        ValueError: If ``method_name`` is not found in the config.
    """
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
    """Generate a human-readable name for a Ray Tune trial.

    Args:
        trial: The Ray Tune :class:`Trial` object.

    Returns:
        A string combining evaluated parameter values and the trial ID.
    """
    params_str = "_".join(
        f"{k}-{v}" for k, v in trial.evaluated_params.items())
    return f"{params_str}_{trial.trial_id}"


def get_tune_param_space(config: dict):
    """Convert a tune-run config dict into a Ray Tune parameter space.

    Supports ``grid_search`` and ``random_search`` strategies. Also promotes
    ``snapshot_config.window_size`` lists into a grid search entry.

    Args:
        config: A merged tune-run config as returned by :func:`get_tune_run_config`.

    Returns:
        A dict mapping parameter names to Ray Tune search objects or fixed values.

    Raises:
        ValueError: If ``search_space`` is not a list.
        NotImplementedError: If an unknown ``search_strategy`` is encountered.
    """
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
    """Restore a model from a Ray Tune checkpoint.

    Args:
        checkpoint: The :class:`ray.tune.Checkpoint` to restore from.
        model_class: The concrete :class:`BaseADModel` subclass to instantiate.
        model_params: Additional keyword arguments forwarded to ``model_class.load``.
        device: Device to map the restored model onto.

    Returns:
        The restored model instance.
    """
    with checkpoint.as_directory() as checkpoint_dir:
        return model_class.load(checkpoint_dir, device, **model_params)
