from __future__ import annotations

import logging
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, field
from typing import Generic, Optional, Self, TypeVar

import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import LinkLoader, NodeLoader

from src.dgadb.storage import TemporalGraph, TemporalGraphSnapshot, TemporalGraphSnapshotLoader
from src.dgadb.experiment.callbacks import ExperimentCallback, ExperimentCallbackHandler


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
class TrainingState:
    epoch: int = 0
    step_in_epoch: int = 0
    total_steps: int = 0
    model: Optional[BaseADModel] = None
    loss: Optional[float] = None
    val_metrics: dict = field(default_factory=dict)


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


BaseADModelComponentsType = TypeVar(
    "BaseADModelComponentsType", bound=BaseADModelComponents)


class BaseADModel(Generic[BaseADModelComponentsType], ABC):
    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = device
        self._components: Optional[BaseADModelComponentsType] = None

    @property
    def components(self) -> BaseADModelComponentsType:
        if self._components is None:
            raise RuntimeError(
                "Model is not initialized. Call `setup(data)` first.")
        return self._components

    def set_training_mode(self, is_training: bool):
        for field in fields(self.components):
            attr = getattr(self.components, field.name)
            if isinstance(attr, torch.nn.Module):
                attr.train() if is_training else attr.eval()

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

    def train(
        self,
        epochs: int,
        train_loader: TemporalGraphSnapshotLoader,
        val_loader: Optional[TemporalGraphSnapshotLoader] = None,
        callbacks: Optional[list[ExperimentCallback]] = None
    ):
        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        for epoch in tqdm(range(epochs)):
            self.set_training_mode(True)
            state.epoch = epoch
            handler.on_train_epoch_begin(state)

            for i, train_snapshot in enumerate(train_loader):
                state.step_in_epoch = i
                state.total_steps += 1
                handler.on_train_step_begin(state)
                state.loss = self._train_step(train_snapshot)
                handler.on_train_step_end(state)

            if val_loader:
                self.set_training_mode(False)
                val_labels, val_probs = self.run_inference(val_loader)
                val_auc = roc_auc_score(
                    val_labels.cpu().numpy(), val_probs.cpu().numpy())
                state.val_metrics = {'val_auc': val_auc}
                print(val_auc)

            handler.on_train_epoch_end(state)

        handler.on_train_end(state)

    @abstractmethod
    def _train_step(self, snapshot: TemporalGraphSnapshot) -> float:
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
