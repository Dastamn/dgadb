import hashlib
import yaml
import logging
import os
import time
from .container import GraphDataContainer
from .callbacks.base import Callback
from .callbacks import Cache
from .container import GraphDataContainer
from .steps.base import PipelineStep
from .steps import DataLoader
from . import steps

_BASE_PATH = os.environ["BASE_PATH"]
# TODO @Dastamn: Update to config/dataset
_CONFIG_PATH = os.path.join(_BASE_PATH, "configs")


def _get_pipeline_steps_hash(steps: list[PipelineStep], length: int = 8) -> str:
    signature = "; ".join(repr(step) for step in steps)
    return hashlib.md5(signature.encode()).hexdigest()[:length]


class Pipeline:
    def __init__(self, steps: list[PipelineStep], callbacks: list[Callback] | None = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.steps = steps
        self.callbacks = callbacks or []

    def run(self, initial_data: GraphDataContainer | None = None) -> GraphDataContainer:
        data = initial_data
        pipeline_start_time = time.time()
        self.logger.info(">>> ---- Starting data pipeline ----")

        for i, step in enumerate(self.steps):
            step_start_time = time.time()
            self.logger.info(f">>> Running step ({i + 1}/{len(self.steps)}): {step.__class__.__name__}")

            # Check for saved state
            cached_result = None
            for callback in self.callbacks:
                result = callback.on_step_begin(step, i, data)
                if result is not None:
                    # A callback returned cached data. We use it and skip this step
                    cached_result = result
                    self.logger.info(f">>> Step '{step.__class__.__name__}' was skipped by loading a cached result.")
                    break  # Stop checking other callbacks for this step

            if cached_result is not None:
                data = cached_result
                continue  # Move to the next step

            # If no cached state was loaded, run the step normally
            try:
                data = step(data)

                # Step end, This runs only if the step was successfully executed
                for callback in self.callbacks:
                    callback.on_step_end(step, i, data)

                step_duration = time.time() - step_start_time
                self.logger.info(f">>> Step '{step.__class__.__name__}' completed in {step_duration:.2f} sec.")
            except Exception as e:
                self.logger.critical(f">>> Pipeline execution failed at step '{step.__class__.__name__}'. Reason: {e}")
                raise

        total_duration = time.time() - pipeline_start_time
        self.logger.info(f">>> ---- Pipeline execution finished successfully in {total_duration:.2f} sec. ----")

        assert data is not None
        return data

    @classmethod
    def from_config(cls, config_name: str, force_rerun: bool = False):
        logger = logging.getLogger(cls.__name__)
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        config_path = os.path.join(_CONFIG_PATH, config_name)
        logger.info(f"Loading config: '{config_path}'")

        if not os.path.exists(config_path):
            error = FileNotFoundError(f"Config file not found: '{config_path}'")
            logger.error(error)
            raise error

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config '{config_path}': {e}")
            raise

        if "pipeline" not in config or "steps" not in config["pipeline"]:
            error = ValueError("Config must contain a 'pipeline.steps' section.")
            logger.error(error)
            raise error

        logger.info("Building preprocessing pipeline from config...")

        pipeline_steps: list[PipelineStep] = []

        for i, step_config in enumerate(config["pipeline"]["steps"]):
            step_name = step_config["name"]
            step_params = step_config.get("params", {})

            if i == 0 and step_name != DataLoader.__name__:
                logger.info(f"> Instantiating missing '{DataLoader.__name__}' step with default parameters.")

                if (
                    "dataset" not in config
                    and "paths" not in config["dataset"]
                    and "structured" not in config["dataset"]["paths"]
                ):
                    error = ValueError("Config must contain a 'dataset.paths.structured' section.")
                    logger.error(error)
                    raise error

                pipeline_steps.append(DataLoader(config["dataset"]["paths"]["structured"]))

            logger.info(f"> Instantiating step {i + 1}: {step_name}")
            try:
                step_class = getattr(steps, step_name)
            except AttributeError:
                logger.error(f"Pipeline step '{step_name}' not found in 'src.preprocessing.step' module.")
                raise ImportError(f"Cannot find step class: '{step_name}'")

            step_is_instanciated = False
            try:
                step_instance = step_class(**step_params)
                step_is_instanciated = True
            except TypeError as e:
                # Try to pass the params as dict, works for FeatureNormalize step
                try:
                    step_instance = step_class(step_params)
                    step_is_instanciated = True
                except TypeError as e:
                    logger.error(f"Mismatched parameters for step '{step_name}'. Check config file: '{config_path}'")
                    logger.error(e)
                    raise

                if not step_is_instanciated:
                    logger.error(f"Mismatched parameters for step '{step_name}'. Check config file: '{config_path}'")
                    logger.error(e)
                    raise

            if not isinstance(step_instance, PipelineStep):
                raise TypeError(f"Class '{step_name}' is not a valid subclass of 'PipelineStep'.")

            pipeline_steps.append(step_instance)

        pipeline_callbacks: list[Callback] = []

        # TODO @Dastamn: Add support for callbacks in config
        pipeline_cache_dir = config["pipeline"].get("cache_dir", None)
        if pipeline_cache_dir is not None:
            logger.info(f"Found cache directory: '{pipeline_cache_dir}'")
            steps_hash = _get_pipeline_steps_hash(pipeline_steps)
            pipeline_cache_dir = os.path.join(pipeline_cache_dir, steps_hash)
            cache_cb = Cache(pipeline_cache_dir, force_rerun)
            pipeline_callbacks.append(cache_cb)

        logger.info("Pipeline built successfully.")

        return Pipeline(pipeline_steps, pipeline_callbacks)

    def __repr__(self) -> str:
        steps_ = "; ".join(repr(step) for step in self.steps)
        return f"{self.__class__.__name__}({steps_})"
