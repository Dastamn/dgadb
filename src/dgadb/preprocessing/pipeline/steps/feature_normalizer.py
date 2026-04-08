from typing import Optional
from .base import PipelineStep
from ..container import GraphDataContainer
from ...normalization.utils import _NormalizerSpec
from ...normalization import prepare_normalizer


class FeatureNormalizer(PipelineStep):
    """Pipeline step that fits and applies feature normalizers.

    Each normalizer is fitted on the training split only and then applied
    to the full edge or node DataFrame, preventing leakage from val/test.

    Args:
        normalizer_configs: List of dict configs converted to specs via
            :func:`prepare_normalizer`.
        normalizer_specs: Pre-built normalizer specs ``(normalizer, columns,
            target)`` to apply directly.
    """

    def __init__(
        self, normalizer_configs: Optional[list[dict]] = None, normalizer_specs: Optional[list[_NormalizerSpec]] = None
    ) -> None:
        super().__init__()
        self.normalizer_configs = normalizer_configs if normalizer_configs else None
        self.normalizer_specs = normalizer_specs if normalizer_specs is not None else []

    def validate(self, data: GraphDataContainer | None) -> None:
        """Verify that the data exists, has been split, and normalizers are configured.

        Args:
            data: The container from the previous step.

        Raises:
            ValueError: If ``data`` is ``None``, not yet split, or no normalizer
                specs have been provided.
        """
        if data is None:
            raise ValueError("Input data is None.")

        if not data.is_split:
            raise ValueError("Data must be split.")

        if not self.normalizer_configs and not self.normalizer_specs:
            raise ValueError("No normalizers provided.")

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        """Fit each normalizer on the train split and apply it to the full DataFrame.

        Returns:
            The container with normalised edge and/or node feature columns.
        """
        assert data is not None
        self.logger.info(
            "Starting feature normalization for nodes and edges...")

        normalizer_specs = self.normalizer_specs
        if self.normalizer_configs:
            normalizer_specs.extend([prepare_normalizer(config)
                                    for config in self.normalizer_configs])

        for normalizer, columns, target in normalizer_specs:
            self.logger.info(
                f"Applying '{normalizer.__class__.__name__}' to {target} features: {columns}")

            train_data = data.train_data

            if target == "edge" or target is None:
                # Defaults to edge if no target is set
                train_df = train_data.edges
                df_to_normalize = data.edges
            elif target == "node":
                train_df = train_data.nodes
                df_to_normalize = data.nodes

                if train_df is None or df_to_normalize is None:
                    self.logger.warning(
                        f"Target is 'node' but no node data found. Skipping.")
                    continue

            else:
                error = NotImplementedError(
                    f"Unknown target provided: '{target}'")
                self.logger.error(error)
                raise

            normalizer.fit(train_df, columns)
            transformed_df = normalizer.transform(df_to_normalize)

            if target == "edge":
                data.edges = transformed_df
            elif target == "node":
                data.nodes = transformed_df

        return data

    def update_metadata(self, data: GraphDataContainer) -> None:
        """Set ``is_feature_normalized = True`` in container metadata."""
        data.is_feature_normalized = True
