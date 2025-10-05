from sklearn.metrics import roc_auc_score
from src.dgadb.evaluation import evaluate
from src.dgadb.storage import TemporalGraphView
from dataclasses import dataclass, fields
from src.dgadb.storage import TemporalGraphSnapshot, TemporalGraphSnapshotLoader
import logging
import typing
from abc import ABC, abstractmethod

import torch
from torch_geometric.loader import LinkLoader, NodeLoader

from src.dgadb.storage import TemporalGraph, TemporalGraphSnapshotLoader
from typing import Callable, Generic, Optional, Self, TypeVar


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
            name
            for name, hint in hints.items()
            if typing.get_origin(hint) is typing.Union and type(None) in typing.get_args(hint)
        ]

        missing = [name for name in required if getattr(self, name) is None]

        if missing:
            raise RuntimeError(
                f"Missing initialization for: {', '.join(missing)}. Call setup() before train() or inference()."
            )

    @abstractmethod
    def setup(self, temporal_graph: TemporalGraph): ...

    @abstractmethod
    def train(self, runnable): ...

    @torch.no_grad()
    @abstractmethod
    def test(
        self, loader: typing.Optional[TemporalGraphSnapshotLoader | LinkLoader | NodeLoader] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class BaseADModelComponents:
    pass

    def to(self, device: torch.device | str):
        for field in fields(self):
            attr = getattr(self, field.name)
            if isinstance(attr, torch.nn.Module) or (isinstance(attr, torch.Tensor) and attr.is_floating_point()):
                setattr(self, field.name, attr.to(device))
            elif isinstance(attr, torch.optim.Optimizer):
                for state in attr.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
        return self


BaseADModelComponentsType = TypeVar("BaseADModelComponentsType", bound=BaseADModelComponents)


class BaseADModel(Generic[BaseADModelComponentsType], ABC):
    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = device
        self._components: Optional[BaseADModelComponentsType] = None

    @property
    def components(self) -> BaseADModelComponentsType:
        if self._components is None:
            raise RuntimeError("Model is not initialized. Call `setup(data)` first.")
        return self._components

    def run_inference(self, loader: TemporalGraphSnapshotLoader) -> tuple[torch.Tensor, torch.Tensor]:
        all_scores = []
        all_labels = []
        for snapshot in loader:
            current_graph = snapshot.current
            edge_scores = self.predict(snapshot)
            edge_labels = current_graph.edge_labels
            all_scores.append(edge_scores)
            all_labels.append(edge_labels)

        all_labels = torch.cat(all_labels)
        all_scores = torch.cat(all_scores)

        return all_labels, all_scores

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        epochs: int,
        train_loader: TemporalGraphSnapshotLoader,
        val_loader: Optional[TemporalGraphSnapshotLoader] = None,
        report_callback: Optional[Callable] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, snapshot: TemporalGraphSnapshot) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def save(self, save_dir: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu") -> Self:
        raise NotImplementedError
