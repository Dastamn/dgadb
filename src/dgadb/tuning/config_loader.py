import os
import yaml
from copy import deepcopy


def deep_update(input_dict: dict, update_dict: dict):
    for k, v in update_dict.items():
        if isinstance(v, dict):
            input_dict[k] = deep_update(input_dict.get(k, {}), v)
        else:
            input_dict[k] = v
    return input_dict


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: '{config_path}'")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(
            f"Error parsing YAML config '{config_path}': {e}")


def get_run_config(config: dict, method_name: str, dataset_name: str) -> dict:
    actual_method_name = None
    for key in config.get('methods', {}).keys():
        if key.lower() == method_name.lower():
            actual_method_name = key
            break
    if not actual_method_name:
        raise ValueError(
            f"Method '{method_name}' not found in config file.")

    # global
    final_config = deepcopy(config.get('global_settings', {}))

    # dataset
    dataset_config = config.get('datasets', {}).get(dataset_name, {})
    final_config = deep_update(final_config, dataset_config)

    # method
    method_config = config['methods'][actual_method_name]
    final_config = deep_update(final_config, method_config)

    # method override
    override_config = method_config.get(
        'overrides', {}).get(dataset_name, {})
    final_config = deep_update(final_config, override_config)

    final_config.pop('overrides', None)

    return final_config


if __name__ == "__main__":
    conf = load_config("configs/tune/tune.yaml")
    run_conf = get_run_config(conf, "strgnn", "as-topology")
    print(run_conf)
    # from dgadb.preprocessing import Pipeline
    # from dgadb.preprocessing.pipeline.steps import TemporalSplitter
    # pipeline = Pipeline.from_config("bitcoin-alpha")
    # step = TemporalSplitter()
    # pipeline.add_step(step)
    # print(pipeline)
