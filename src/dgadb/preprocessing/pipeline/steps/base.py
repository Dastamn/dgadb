import logging
from abc import ABC, abstractmethod
from ..container import GraphDataContainer


class PipelineStep(ABC):
    """Abstract base class for a single preprocessing step.

    Subclasses implement :meth:`validate`, :meth:`process` and
    :meth:`update_metadata`. The orchestrating :class:`Pipeline` invokes
    instances via ``__call__``, which wraps validation and processing.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, data: GraphDataContainer | None) -> None:
        """Check that ``data`` meets this step's prerequisites.

        Args:
            data: The container produced by the previous step, or ``None``
                for the first step in the pipeline.

        Raises:
            ValueError: If the container does not satisfy the step's
                preconditions.
        """
        ...

    @abstractmethod
    def process(self, data: GraphDataContainer | None) -> "GraphDataContainer":
        """Execute the core logic of this step and return the updated container.

        Args:
            data: The container from the previous step, or ``None`` for the
                first step (which is expected to construct the initial container).

        Returns:
            A new or updated :class:`GraphDataContainer`.
        """
        ...

    @abstractmethod
    def update_metadata(self, data: GraphDataContainer) -> None:
        """Write step-specific flags and statistics into ``data.metadata``.

        Args:
            data: The container returned by :meth:`process`.
        """
        ...

    def __call__(self, data: GraphDataContainer | None) -> "GraphDataContainer":
        """Validate, process, and update metadata — called by the Pipeline orchestrator.

        Args:
            data: The container from the previous step, or ``None`` for the
                first step.

        Returns:
            The container produced by :meth:`process` with metadata updated.
        """
        try:
            self.validate(data)
        except Exception as e:
            self.logger.error(f"Validation failed for step {self.__class__.__name__}: {e}")
            raise

        processed_data = self.process(data)
        self.update_metadata(processed_data)

        return processed_data

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if k != "logger")
        return f"{self.__class__.__name__}({params})"
