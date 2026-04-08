import os
import yaml
import logging
from dgadb.utils.paths import get_project_root

logger = logging.getLogger(__name__)


def load_config(name: str) -> dict:
    """Load a named config YAML from the project's ``configs/`` directory.

    Args:
        name: Config name without the ``.yaml`` extension (e.g. ``"tune"``).

    Returns:
        Parsed config as a plain dict.

    Raises:
        FileNotFoundError: If ``configs/<name>.yaml`` does not exist.
    """
    base_path = get_project_root()
    yaml_path = os.path.join(base_path, "configs", f"{name}.yaml")

    logger.info(f"Loading config: {yaml_path}")

    if not os.path.exists(yaml_path):
        logger.error(f"Config file does not exist: {yaml_path}")
        raise FileNotFoundError(f"Config file does not exist: {yaml_path}")

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config: {name}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config '{yaml_path}': {e}")
        raise
