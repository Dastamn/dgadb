import os
import yaml

def load_config(name: str) -> dict:
    base_path = os.environ["BASE_PATH"]
    yaml_path = os.path.join(base_path, "configs", f"{name}.yaml")

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)