from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import torch
from aim import Run

from .base import ExperimentCallback

if TYPE_CHECKING:
    from dgadb.models.base import TrainingState


class AimCallback(ExperimentCallback):
    """A callback to track experiments with Aim.

    This callback provides comprehensive experiment tracking including:
    - Hyperparameters and model configuration
    - Per-step loss tracking
    - Per-epoch validation metrics
    - Experiment metadata (dataset, anomaly config, etc.)

    Args:
        experiment_name: Name for the experiment group in Aim.
        repo: Path to the Aim repository. If None, uses the default `.aim` directory.
        run_name: Optional name for this specific run.
        hparams: Dictionary of hyperparameters to log.
        tags: List of tags to attach to the run.
        log_system_metrics: Whether to log system metrics (CPU, memory, etc.).
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        repo: str | None = None,
        run_name: str | None = None,
        hparams: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        log_system_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.experiment_name = experiment_name
        self.repo = repo
        self.run_name = run_name
        self.hparams = hparams or {}
        self.tags = tags or []
        self.log_system_metrics = log_system_metrics
        self.run: Run | None = None
        self._pending_configs: dict[str, dict[str, Any]] = {}

    def on_train_begin(self, state: TrainingState) -> None:
        self.run = Run(
            experiment=self.experiment_name,
            repo=self.repo,
            log_system_params=self.log_system_metrics,
        )

        if self.run_name:
            self.run.name = self.run_name

        for tag in self.tags:
            self.run.add_tag(tag)

        # Log hyperparameters
        if self.hparams:
            self.run["hparams"] = self.hparams

        # Log any pending configs that were set before training began
        for config_name, config_value in self._pending_configs.items():
            self.run[config_name] = config_value
        self._pending_configs.clear()

        # Log model information
        if state.model is not None:
            model_info = {
                "model_class": state.model.__class__.__name__,
                "device": str(state.model.device),
            }

            # Extract model components info if available
            if state.model._components is not None:
                components_info = {}
                for field in fields(state.model._components):
                    component = getattr(state.model._components, field.name)
                    if hasattr(component, "__class__"):
                        components_info[field.name] = component.__class__.__name__
                    # Count parameters for nn.Module components (skip uninitialized lazy params)
                    if hasattr(component, "parameters"):
                        try:
                            param_count = sum(
                                p.numel() for p in component.parameters()
                                if not isinstance(p, torch.nn.UninitializedParameter)
                            )
                            components_info[f"{field.name}_params"] = param_count
                        except ValueError:
                            # LazyModule parameters not yet initialized
                            components_info[f"{field.name}_params"] = "uninitialized"
                model_info["components"] = components_info

            self.run["model"] = model_info

        self.logger.info(f"Aim tracking initialized: experiment={self.experiment_name}")

    def on_train_step_end(self, state: TrainingState) -> None:
        if self.run is None:
            return

        # Log loss at step granularity
        if state.loss is not None:
            self.run.track(
                state.loss,
                name="loss",
                step=state.total_steps,
                epoch=state.epoch,
                context={"subset": "train"},
            )

    def on_train_epoch_end(self, state: TrainingState) -> None:
        if self.run is None:
            return

        # Log validation metrics
        for metric_name, metric_value in state.val_metrics.items():
            self.run.track(
                metric_value,
                name=metric_name,
                epoch=state.epoch,
                context={"subset": "val"},
            )

        # Log epoch summary
        self.run.track(
            state.total_steps,
            name="total_steps",
            epoch=state.epoch,
        )

    def on_train_end(self, state: TrainingState) -> None:
        if self.run is not None:
            self.run.close()
            self.logger.info("Aim tracking finalized")

    def log_evaluation_metrics(self, metrics: dict[str, float], context: str = "test") -> None:
        """Log evaluation metrics from ADEvaluator.

        This method can be called externally to log final evaluation metrics.

        Args:
            metrics: Dictionary of metric names to values.
            context: Context label for the metrics (e.g., "test", "final").
        """
        if self.run is None:
            self.logger.warning("Aim run not initialized. Call on_train_begin first.")
            return

        for metric_name, metric_value in metrics.items():
            self.run.track(
                metric_value,
                name=metric_name,
                context={"subset": context},
            )

    def log_config(self, config: dict[str, Any], name: str = "config") -> None:
        """Log arbitrary configuration dictionaries.

        Can be called before or after training begins. If called before,
        configs are stored and logged when the run starts.

        Args:
            config: Configuration dictionary to log.
            name: Name for the configuration in Aim.
        """
        if self.run is None:
            # Store for later logging when run is initialized
            self._pending_configs[name] = config
        else:
            self.run[name] = config
