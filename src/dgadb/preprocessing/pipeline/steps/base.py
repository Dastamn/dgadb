import logging
from abc import ABC, abstractmethod
from ..container import GraphDataContainer


class PipelineStep(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, data: GraphDataContainer | None) -> None:
        """
        Validates that the input data meets the prerequisites for this step.
        The first step in the pipeline will receive `None`.
        """
        ...

    @abstractmethod
    def process(self, data: GraphDataContainer | None) -> "GraphDataContainer":
        """
        Executes the core logic of the step.
        The first step in the pipeline will receive `None` and is expected to create
        the initial GraphDataContainer.
        """
        ...

    @abstractmethod
    def update_metadata(self, data: GraphDataContainer) -> None:
        """
        Updates GraphDataContainer metadata.
        """
        ...

    def __call__(self, data: GraphDataContainer | None) -> "GraphDataContainer":
        """
        The main execution method called by the Pipeline orchestrator.
        It wraps validation and processing.
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
