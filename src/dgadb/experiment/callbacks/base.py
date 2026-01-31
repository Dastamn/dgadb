from __future__ import annotations

import logging
from abc import ABC
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models.base import TrainingState


class ExperimentCallback(ABC):
    """Abstract base class for creating callbacks.

    Callbacks are objects that can perform actions at various stages of the
    training loop. They provide a way to add custom functionality like logging,
    model checkpointing, or resource monitoring without modifying the core
    training logic in `BaseADModel`.

    To create a custom callback, inherit from this class and override the
    methods corresponding to the events you want to handle.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbose = verbose

    def on_train_begin(self, state: TrainingState):
        pass

    def on_train_end(self, state: TrainingState):
        pass

    def on_train_epoch_begin(self, state: TrainingState):
        pass

    def on_train_epoch_end(self, state: TrainingState):
        pass

    def on_train_step_begin(self, state: TrainingState):
        pass

    def on_train_step_end(self, state: TrainingState):
        pass


class ExperimentCallbackHandler:
    """Manages and executes a list of `ExperimentCallback` instances.

    This class acts as a dispatcher, iterating through a list of registered
    callbacks and calling their corresponding hook methods at the appropriate
    times during the training loop.

    Attributes:
        callbacks: A list of `ExperimentCallback` objects.
    """

    def __init__(self, callbacks: Optional[list[ExperimentCallback]] = None) -> None:
        self.callbacks = callbacks or []

    def on_train_begin(self, state: TrainingState):
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: TrainingState):
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_train_epoch_begin(self, state: TrainingState):
        for cb in self.callbacks:
            cb.on_train_epoch_begin(state)

    def on_train_epoch_end(self, state: TrainingState):
        for cb in self.callbacks:
            cb.on_train_epoch_end(state)

    def on_train_step_begin(self, state: TrainingState):
        for cb in self.callbacks:
            cb.on_train_step_begin(state)

    def on_train_step_end(self, state: TrainingState):
        for cb in self.callbacks:
            cb.on_train_step_end(state)
