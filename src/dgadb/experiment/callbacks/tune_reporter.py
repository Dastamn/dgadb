from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

from ray import tune
from ray.tune import Checkpoint

from .base import ExperimentCallback

if TYPE_CHECKING:
    from src.dgadb.models.base import TrainingState


class TuneReporter(ExperimentCallback):
    """A callback to report metrics and checkpoints to Ray Tune.

    This callback integrates the training loop with Ray Tune for hyperparameter
    optimization. At the end of each epoch, it reports validation metrics to
    Tune. It can also periodically save model checkpoints.

    Args:
        checkpoint_epoch_interval: The frequency (in epochs) at which to save
            a model checkpoint and report it to Ray Tune.
    """

    def __init__(self, checkpoint_epoch_interval: int = 5) -> None:
        super().__init__()
        self.checkpoint_epoch_interval = checkpoint_epoch_interval

    def on_train_epoch_end(self, state: TrainingState):
        if not state.val_metrics:
            return

        checkpoint = None
        if state.model is not None and state.epoch % self.checkpoint_epoch_interval == 0:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                state.model.save(checkpoint_dir)
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                tune.report(metrics=state.val_metrics, checkpoint=checkpoint)
        else:
            tune.report(metrics=state.val_metrics)
