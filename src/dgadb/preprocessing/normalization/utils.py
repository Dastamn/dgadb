import logging
from typing import NamedTuple, Literal, Optional
from .base import Normalizer
from .. import normalization

logger = logging.getLogger(__name__)


class _NormalizerSpec(NamedTuple):
    normalizer: Normalizer
    columns: list[str]
    target: Optional[Literal["node", "edge"]]


def prepare_normalizer(normalizer_config: dict) -> _NormalizerSpec:
    name = normalizer_config["name"]
    target = normalizer_config.get("target", None)
    columns = normalizer_config.get("columns", [])
    params = normalizer_config.get("params", {})

    try:
        normalizer_class = getattr(normalization, name)
    except AttributeError:
        logger.error(f"Normalizer '{name}' not found in 'src.preprocessing.normalization' module.")
        raise ImportError(f"Cannot find step class: '{name}'")

    try:
        normalizer_instance = normalizer_class(**params)
    except TypeError as e:
        logger.error(f"Mismatched parameters for step '{name}'.")
        logger.error(e)
        raise

    if not isinstance(normalizer_instance, Normalizer):
        raise TypeError(f"Class '{name}' is not a valid subclass of 'Normalizer'.")

    return _NormalizerSpec(normalizer_instance, columns, target)
