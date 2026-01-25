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

from dgadb.storage import TemporalGraph, TemporalGraphSnapshot, TemporalGraphSnapshotLoader
from dgadb.experiment.callbacks import ExperimentCallback, ExperimentCallbackHandler


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
    """Holds the current state of the training loop.

    This dataclass is used to pass information about the training progress
    to callbacks, allowing them to inspect and react to the state of the model
    and the training process at various points.

    Attributes:
        epoch: The current epoch number, starting from 0.
        step_in_epoch: The current step number within the current epoch.
        total_steps: The total number of training steps completed so far.
        model: A reference to the model instance being trained.
        loss: The loss value computed in the most recent training step.
        val_metrics: A dictionary of metrics computed during the last
            validation run.
    """

    epoch: int = 0
    step_in_epoch: int = 0
    total_steps: int = 0
    model: Optional[BaseADModel] = None
    loss: Optional[float] = None
    val_metrics: dict = field(default_factory=dict)


@dataclass
class BaseADModelComponents:
    """Base class for a dataclass holding model components.

    This class serves as a container for all trainable or device-dependent
    parts of a model, such as `torch.nn.Module` instances, optimizers, and
    tensors. Inheriting from this class and defining components as fields
    allows for easy management, particularly for moving all components to a
    specific device using the `to` method.

    Example:
        @dataclass
        class MyModelComponents(BaseADModelComponents):
            encoder: torch.nn.Module
            decoder: torch.nn.Module
            optimizer: torch.optim.Optimizer
    """

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
    """Abstract base class for anomaly detection models.

    This class provides a standardized framework for setting up, training,
    and evaluating anomaly detection models on temporal graphs. It manages the
    training loop and integrates with a callback system for extensibility.

    To create a new model, inherit from this class and implement the abstract
    methods: `setup`, `_train_step`, `_predict`, `save`, and `load`.

    Arguments:
        device: The torch device ('cpu' or 'cuda') on which the model runs.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self._components: Optional[BaseADModelComponentsType] = None

    @property
    def components(self) -> BaseADModelComponentsType:
        """Provides access to the model's components.
        Should be used instead of calling `self._components`to ensure
        model initialization.

        Raises:
            RuntimeError: If the model has not been initialized via `setup()`.
        """
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
        for snapshot in tqdm(loader, desc="TEST"):
            current_graph = snapshot.current
            edge_scores = self._predict(snapshot)
            edge_labels = current_graph.edge_labels
            all_scores.append(edge_scores)
            all_labels.append(edge_labels)

        all_labels = torch.cat(all_labels)
        all_scores = torch.cat(all_scores)

        return all_labels, all_scores

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        """Initializes the model and its components.

        This method should be implemented by subclasses to prepare the model for
        training based on the structure of the input temporal graph data. This
        typically involves instantiating neural network layers with correct
        dimensions and setting up the optimizer.

        Args:
            data: The temporal graph dataset used for initialization.
            **kwargs: Additional arguments for setup.
        """
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

        for epoch in range(epochs):
            self.set_training_mode(True)
            state.epoch = epoch
            handler.on_train_epoch_begin(state)

            for i, train_snapshot in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"TRAIN - Epoch {epoch+1}/{epochs}"):
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
                state.val_metrics = {'roc_auc': val_auc}
                print(val_auc)

            handler.on_train_epoch_end(state)

        handler.on_train_end(state)

    @abstractmethod
    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def save(self, save_dir: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> Self:
        raise NotImplementedError
