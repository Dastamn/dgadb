import logging
import typing
from abc import ABC, abstractmethod

import torch

from src.dgadb.storage import TemporalGraph


class BaseModel(ABC):
    temporal_graph: typing.Optional[TemporalGraph]
    model: typing.Optional[torch.nn.Module]
    optimizer: typing.Optional[torch.optim.Optimizer]

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.temporal_graph = None
        self.model = None
        self.optimizer = None

    def _ensure_setup(self):
        hints = typing.get_type_hints(BaseModel)
        required = [
            name for name, hint in hints.items()
            if typing.get_origin(hint) is typing.Union
            and type(None) in typing.get_args(hint)
        ]

        missing = [name for name in required if getattr(self, name) is None]

        if missing:
            raise RuntimeError(
                f"Missing initialization for: {', '.join(missing)}. "
                "Call setup() before train() or inference()."
            )

    @abstractmethod
    def setup(self, temporal_graph: TemporalGraph):
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def inference(self):
        ...
