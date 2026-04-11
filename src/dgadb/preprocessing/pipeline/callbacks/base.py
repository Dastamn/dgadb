from ..container import GraphDataContainer
from ..steps.base import PipelineStep


class Callback:
    """Base class for custom actions to be executed during pipeline execution."""

    def on_step_begin(
        self, step: PipelineStep, step_index: int, data: GraphDataContainer | None
    ) -> GraphDataContainer | None:
        """Called before a step executes; may return a cached container to skip the step.

        Args:
            step: The step about to run.
            step_index: Zero-based position of the step in the pipeline.
            data: The container from the preceding step (or ``None`` for the first step).

        Returns:
            A cached :class:`GraphDataContainer` to skip execution, or ``None``
            to let the step run normally.
        """
        return None

    def on_step_end(self, step: PipelineStep, step_index: int, data: GraphDataContainer):
        """Called after a step completes successfully.

        Args:
            step: The step that just finished.
            step_index: Zero-based position of the step in the pipeline.
            data: The container returned by the step.
        """
        pass
