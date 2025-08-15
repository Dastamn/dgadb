from ..container import GraphDataContainer
from ..steps.base import PipelineStep


class Callback:
    """Base class for custom actions to be executed during pipeline execution."""

    def on_step_begin(
        self, step: PipelineStep, step_index: int, data: GraphDataContainer | None
    ) -> GraphDataContainer | None:
        """
        Called before a step starts. Can optionally return a cached data object
        to make the pipeline skip the current step's execution.
        """
        return None

    def on_step_end(self, step: PipelineStep, step_index: int, data: GraphDataContainer):
        """Called after a pipeline step is successfully executed."""
        pass
