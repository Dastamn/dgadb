"""Callback base class and dispatch handler for the training loop.

The training loop in :class:`~dgadb.models.base.BaseADModel` invokes a
list of :class:`ExperimentCallback` instances at six well-defined hook
points (``on_train_begin``, ``on_train_end``, ``on_train_epoch_begin``,
``on_train_epoch_end``, ``on_train_step_begin``, ``on_train_step_end``).
Subclassing :class:`ExperimentCallback` and overriding any subset of
these methods is the supported way to add custom logging, model
checkpointing, resource monitoring, or hyperparameter-tuning hooks
without touching the training loop itself.

The concrete callbacks shipped with dgadb live in sibling modules:

* :class:`~dgadb.experiment.callbacks.aim_tracker.AimCallback`
* :class:`~dgadb.experiment.callbacks.resource_monitor.ResourceMonitor`
* :class:`~dgadb.experiment.callbacks.tune_reporter.TuneReporter`

:class:`ExperimentCallbackHandler` is the dispatcher that walks a list
of registered callbacks at each hook point; users do not normally
construct it directly — :class:`~dgadb.models.base.BaseADModel.train`
does that internally from the ``callbacks=`` argument.
"""

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
        """Called once before the first training epoch starts.

        Args:
            state: Current training state snapshot.
        """
        pass

    def on_train_end(self, state: TrainingState):
        """Called once after the last training epoch finishes.

        Args:
            state: Final training state snapshot.
        """
        pass

    def on_train_epoch_begin(self, state: TrainingState):
        """Called at the start of each training epoch.

        Args:
            state: Current training state snapshot.
        """
        pass

    def on_train_epoch_end(self, state: TrainingState):
        """Called at the end of each training epoch.

        Args:
            state: Current training state snapshot including validation metrics.
        """
        pass

    def on_train_step_begin(self, state: TrainingState):
        """Called before each individual training step (snapshot).

        Args:
            state: Current training state snapshot.
        """
        pass

    def on_train_step_end(self, state: TrainingState):
        """Called after each individual training step (snapshot).

        Args:
            state: Current training state snapshot including the step loss.
        """
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
        """Dispatch ``on_train_begin`` to all registered callbacks."""
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: TrainingState):
        """Dispatch ``on_train_end`` to all registered callbacks."""
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_train_epoch_begin(self, state: TrainingState):
        """Dispatch ``on_train_epoch_begin`` to all registered callbacks."""
        for cb in self.callbacks:
            cb.on_train_epoch_begin(state)

    def on_train_epoch_end(self, state: TrainingState):
        """Dispatch ``on_train_epoch_end`` to all registered callbacks."""
        for cb in self.callbacks:
            cb.on_train_epoch_end(state)

    def on_train_step_begin(self, state: TrainingState):
        """Dispatch ``on_train_step_begin`` to all registered callbacks."""
        for cb in self.callbacks:
            cb.on_train_step_begin(state)

    def on_train_step_end(self, state: TrainingState):
        """Dispatch ``on_train_step_end`` to all registered callbacks."""
        for cb in self.callbacks:
            cb.on_train_step_end(state)
