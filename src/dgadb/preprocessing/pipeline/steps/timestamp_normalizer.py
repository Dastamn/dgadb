import polars as pl
from .base import PipelineStep
from ..container import GraphDataContainer


class TimestampNormalizer(PipelineStep):
    """Pipeline step that casts edge (and node) timestamps to ``UInt64``.

    Date/datetime columns are cast directly; integer-like float columns are
    truncated; floats with fractional parts are scaled by a power of ten so
    that all decimals are preserved before casting.
    """

    def __init__(self) -> None:
        super().__init__()

    def validate(self, data: GraphDataContainer | None) -> None:
        """Check that the container exists and timestamp columns are numeric or temporal.

        Args:
            data: The container from the previous step.

        Raises:
            ValueError: If ``data`` is ``None`` or a timestamp column has an
                unsupported dtype.
        """
        if data is None:
            raise ValueError("Input data is None.")

        edge_timestamps = data.edge_timestamps
        node_timestamps = data.node_timestamps

        if node_timestamps is not None and not (
            node_timestamps.dtype.is_numeric() or node_timestamps.dtype.is_temporal()
        ):
            raise ValueError(
                f"'{data.n_time_col}' column in '{data.__class__.__name__}' is not numerical nor temporal."
            )

        if not (edge_timestamps.dtype.is_numeric() or edge_timestamps.dtype.is_temporal()):
            raise ValueError(
                f"'{data.e_time_col}' column in '{data.__class__.__name__}' is not numerical nor temporal."
            )

    def _normalize_timestamps(self, df: pl.DataFrame, time_col: str) -> pl.DataFrame:
        dtype = df[time_col].dtype
        timestamp_series = df[time_col]
        self.logger.info(
            f"Normalizing timestamp column '{time_col}' with dtype {dtype}.")

        # Date / Datetime
        if dtype in (pl.Date, pl.Datetime):
            normalized_expr = pl.col(time_col).cast(pl.UInt64)

        # Numeric types (integer or float)
        elif dtype.is_numeric():
            # Check if a float column is effectively all integers
            is_integer_like = False
            if dtype.is_float():
                if (timestamp_series != timestamp_series.floor()).sum() == 0:
                    is_integer_like = True

            if dtype.is_integer() or is_integer_like:
                self.logger.info("  - Treating as an integer-like sequence.")
                normalized_expr = pl.col(time_col).cast(pl.UInt64)
            else:
                # Float type
                self.logger.info(
                    "  - Treating as a float with fractional parts. Scaling to preserve precision.")
                max_decimals = (
                    timestamp_series.cast(pl.Utf8).str.extract(
                        r"\.(\d*)$", group_index=1).str.len_chars().max()
                )

                scale_factor = 10 ** (max_decimals or 0)
                self.logger.info(
                    f"  - Scaling factor determined: {scale_factor}")
                normalized_expr = (pl.col(time_col) *
                                   scale_factor).cast(pl.UInt64)

        # Unsupported types
        else:
            raise TypeError(
                f"Timestamp normalization does not support dtype: {dtype}.")

        # Apply the transformation, replacing the original column
        return df.with_columns(normalized_expr.alias(time_col))

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        """Cast edge and node timestamp columns to ``UInt64``.

        Returns:
            The container with integer-valued timestamp columns.
        """
        assert data is not None

        self.logger.info("Starting timestamp normalization...")

        # Normalize timestamps for the edges DataFrame
        data.edges = self._normalize_timestamps(data.edges, data.e_time_col)
        self.logger.info("Successfully normalized timestamps for edges.")

        # Normalize timestamps for the nodes DataFrame, if applicable
        if data.nodes is not None and data.node_timestamps is not None:
            data.nodes = self._normalize_timestamps(
                data.nodes, data.n_time_col)
            self.logger.info("Successfully normalized timestamps for nodes.")

        return data

    def update_metadata(self, data: GraphDataContainer) -> None:
        """Set ``is_timestamp_normalized = True`` in container metadata."""
        data.is_timestamp_normalized = True
