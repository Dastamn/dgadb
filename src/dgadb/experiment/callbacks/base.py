from __future__ import annotations

import logging
from abc import ABC
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models.base import TrainingState


class ExperimentCallback(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

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
