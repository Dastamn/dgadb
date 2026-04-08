import numpy as np
from typing import Optional
import polars as pl
from ..container import GraphDataContainer
from .base import PipelineStep


class TemporalSplitter(PipelineStep):
    """Pipeline step that adds a chronological train/val/test split column.

    Edges are partitioned by timestamp quantiles so that train precedes val
    precedes test in time, preventing temporal leakage.

    Args:
        train_ratio: Fraction of edges (by timestamp quantile) for training.
        val_ratio: Fraction for validation; defaults to ``0.0``.
        split_col: Name of the column to add to ``edges`` holding split labels.

    Raises:
        ValueError: If the supplied ratios are out of range.
    """

    def __init__(self, train_ratio: float = 0.7, val_ratio: Optional[float] = None, split_col: str = "split") -> None:
        super().__init__()
        if val_ratio is None:
            val_ratio = 0.0

        if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
            raise ValueError("Invalid ratios provided.")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.split_col = split_col

    def validate(self, data: GraphDataContainer | None) -> None:
        """Check that the container exists and has a numeric/temporal timestamp column.

        Args:
            data: The container from the previous step.

        Raises:
            ValueError: If ``data`` is ``None`` or the timestamp column has
                an unsupported dtype.
        """
        if data is None:
            raise ValueError("Input data is None.")

        edge_timestamps = data.edge_timestamps
        if not (edge_timestamps.dtype.is_numeric() or edge_timestamps.dtype.is_temporal()):
            raise ValueError(
                f"'{data.e_time_col}' column in '{data.__class__.__name__}' is not numerical nor temporal."
            )

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        """Add a ``split`` column to edges using chronological quantile cutoffs.

        Returns:
            The container with a new split label column on the edges DataFrame.
        """
        assert data is not None
        self.logger.info(f"Total edges: {len(data.edges)}")
        self.logger.info(
            f"Performing chronological split: "
            f"Train={self.train_ratio:.2f}, Val={self.val_ratio:.2f}, Test={self.test_ratio:.2f}"
        )

        # Determine timestamp cutoff times from edges
        edge_timestamps_array = data.edge_timestamps.to_numpy()
        train_cutoff_time, val_cutoff_time = np.quantile(
            edge_timestamps_array,
            [(1 - self.val_ratio - self.test_ratio), (1 - self.test_ratio)],
        )
        self.logger.info(
            f"Train cutoff time: {train_cutoff_time}, Validation cutoff time: {val_cutoff_time}")

        data.edges = data.edges.with_columns(
            pl.when(pl.col(data.e_time_col) <= train_cutoff_time)
            .then(pl.lit("train"))
            .when(pl.col(data.e_time_col) <= val_cutoff_time)
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias(self.split_col)
        )

        vc = data.edges[self.split_col].value_counts()
        split_counts = dict(
            zip(vc[self.split_col].to_list(), vc["count"].to_list()))
        self.logger.info(f"Edge split counts: {split_counts}")

        return data

    def update_metadata(self, data: GraphDataContainer) -> None:
        """Mark the container as split and record split ratios in metadata."""
        data.is_split = True
        data.split_col = self.split_col
        data.splits = {"train_ratio": self.train_ratio,
                       "val_ratio": self.val_ratio, "test_ratio": self.test_ratio}
