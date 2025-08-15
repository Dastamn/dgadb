import polars as pl
from .base import PipelineStep
from ..container import GraphDataContainer


class TimestampNormalizer(PipelineStep):
    def __init__(self) -> None:
        super().__init__()

    def validate(self, data: GraphDataContainer | None) -> None:
        if data is None:
            raise ValueError("Input data is None.")

        edge_timestamps = data.edge_timestamps
        node_timestamps = data.node_timestamps

        if node_timestamps is not None and not (
            node_timestamps.dtype.is_numeric() or node_timestamps.dtype.is_temporal()
        ):
            raise ValueError(
                f"'{data.metadata.n_time_col}' column in '{data.__class__.__name__}' is not numerical nor temporal."
            )

        if not (edge_timestamps.dtype.is_numeric() or edge_timestamps.dtype.is_temporal()):
            raise ValueError(
                f"'{data.metadata.e_time_col}' column in '{data.__class__.__name__}' is not numerical nor temporal."
            )

    def _normalize_timestamps(self, df: pl.DataFrame, timestamp_col: str) -> pl.DataFrame:
        dtype = df[timestamp_col].dtype
        timestamp_series = df[timestamp_col]
        self.logger.info(f"Normalizing timestamp column '{timestamp_col}' with dtype {dtype}.")

        # Date / Datetime
        if dtype in (pl.Date, pl.Datetime):
            normalized_expr = pl.col(timestamp_col).cast(pl.UInt64)

        # Numeric types (integer or float)
        elif dtype.is_numeric():
            # Check if a float column is effectively all integers
            is_integer_like = False
            if dtype.is_float():
                if (timestamp_series != timestamp_series.floor()).sum() == 0:
                    is_integer_like = True

            if dtype.is_integer() or is_integer_like:
                self.logger.info("  - Treating as an integer-like sequence.")
                normalized_expr = pl.col(timestamp_col).cast(pl.UInt64)
            else:
                # Float type
                self.logger.info("  - Treating as a float with fractional parts. Scaling to preserve precision.")
                max_decimals = (
                    timestamp_series.cast(pl.Utf8).str.extract(r"\.(\d*)$", group_index=1).str.len_chars().max()
                )

                scale_factor = 10 ** (max_decimals or 0)
                self.logger.info(f"  - Scaling factor determined: {scale_factor}")
                normalized_expr = (pl.col(timestamp_col) * scale_factor).cast(pl.UInt64)

        # Unsupported types
        else:
            raise TypeError(f"Timestamp normalization does not support dtype: {dtype}.")

        # Apply the transformation, replacing the original column
        return df.with_columns(normalized_expr.alias(timestamp_col))

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        assert data is not None

        self.logger.info("Starting timestamp normalization...")

        # Normalize timestamps for the edges DataFrame
        data.edges = self._normalize_timestamps(data.edges, data.metadata.e_time_col)
        self.logger.info("Successfully normalized timestamps for edges.")

        # Normalize timestamps for the nodes DataFrame, if applicable
        if data.nodes is not None and data.node_timestamps is not None:
            data.nodes = self._normalize_timestamps(data.nodes, data.metadata.n_time_col)
            self.logger.info("Successfully normalized timestamps for nodes.")

        return data

    def update_metadata(self, data: GraphDataContainer) -> None:
        data.set_metadata("is_timestamp_normalized", True)
