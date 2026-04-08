import os
import pickle
import logging
from .base import Callback
from ..container import GraphDataContainer
from ..steps.base import PipelineStep


class Cache(Callback):
    """A callback that saves and loads intermediate pipeline results to/from a cache directory.

    Args:
        cache_dir: Directory where per-step pickle files are stored.
            Created automatically if it does not exist.
        force_rerun: If ``True``, existing cache files are ignored and
            every step is re-executed.
    """

    def __init__(self, cache_dir: str, force_rerun: bool = False):
        self.cache_dir = cache_dir
        self.force_rerun = force_rerun
        self.logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"Created cache directory: '{self.cache_dir}'")
        else:
            self.logger.info(f"Using existing cache directory: '{self.cache_dir}'")

        if self.force_rerun:
            self.logger.warning("'force_rerun' is True. All steps will be re-executed and cache will be overwritten.")

    def _get_cache_path(self, step: PipelineStep, step_index: int) -> str:
        """Generates a consistent file path for a step's cache."""
        step_name = f"{step_index:02d}_{step.__class__.__name__}"
        return os.path.join(self.cache_dir, f"{step_name}.pkl")

    def on_step_begin(
        self, step: PipelineStep, step_index: int, data: GraphDataContainer | None
    ) -> GraphDataContainer | None:
        """Check for a cached result before the step runs."""
        cache_path = self._get_cache_path(step, step_index)

        if os.path.exists(cache_path) and not self.force_rerun:
            self.logger.info(f"Found cached result for '{step.__class__.__name__}'. Loading from '{cache_path}'")
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                return cached_data
            except Exception as e:
                self.logger.error(f"Failed to load cache file '{cache_path}'. Re-running step. Error: {e}")

        return None

    def on_step_end(self, step: PipelineStep, step_index: int, data: GraphDataContainer):
        """Save the result of a step after it has run."""
        cache_path = self._get_cache_path(step, step_index)
        self.logger.info(f"Saving result of '{step.__class__.__name__}' to '{cache_path}'")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

        except Exception as e:
            self.logger.error(f"Failed to save cache file to '{cache_path}'. Error: {e}")
