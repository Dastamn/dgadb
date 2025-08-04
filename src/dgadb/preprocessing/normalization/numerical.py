import numpy as np
import polars as pl
import logging
logger = logging.getLogger(__name__)


class StandardNormalizer:
    """
    A normalizer that standardizes features by removing the mean and scaling to
    unit variance, similar to scikit-learn's StandardScaler.
    """

    def __init__(self) -> None:
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        self.columns: list[str] = []

    def fit(self, df: pl.DataFrame, columns: list[str]) -> 'StandardNormalizer':
        """
        Computes the mean and standard deviation for the specified columns.

        Args:
            df: The Polars DataFrame to fit on.
            columns: A list of column names to normalize.

        Returns:
            The fitted normalizer instance.
        """
        self.columns = columns
        if not self.columns:
            logger.warning("Warning: No columns provided to fit.")
            return self

        stats = df.select(
            [pl.col(c).mean().alias(f"{c}_mean") for c in self.columns]
            + [pl.col(c).std().alias(f"{c}_std") for c in self.columns]
        )
        for col in self.columns:
            self.mean_[col] = stats.select(f"{col}_mean").item()
            std_val = stats.select(f"{col}_std").item()
            # Avoid division by 0 and by very small numbers
            self.std_[col] = std_val if std_val > 1e-9 else 1.0

        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """
        Standardizes the specified columns in the DataFrame.

        Args:
            df: The DataFrame to transform.

        Returns:
            A NumPy array of the transformed data.
        """
        if not self.columns:
            raise RuntimeError(
                "'StandardNormalizer' must be fitted before transform.")

        return df.select([
            ((pl.col(col) - self.mean_[col]) / self.std_[col])
            for col in self.columns
        ]).to_numpy()

    def inverse_transform(self, data: np.ndarray) -> pl.DataFrame:
        """
        Applies the inverse standardization.

        Args:
            data: A NumPy array of standardized data, with columns in the
                  same order as they were fitted.

        Returns:
            A Polars DataFrame of the data scaled back to its original representation.
        """
        if not self.columns:
            raise RuntimeError(
                "'StandardNormalizer' must be fitted before inverse transform.")

        if data.shape[1] != len(self.columns):
            raise ValueError(
                f"Input array has {data.shape[1]} columns, but normalizer was "
                f"fitted on {len(self.columns)} columns."
            )

        df = pl.DataFrame(data, schema=self.columns, strict=False)

        return df.with_columns([
            ((pl.col(col) * self.std_[col]) + self.mean_[col])
            for col in self.columns
        ])

    def get_params(self) -> dict:
        """Returns the learned parameters."""
        return {"mean_": self.mean_, "std_": self.std_, "columns": self.columns}


class MinMaxNormalizer:
    """
    A normalizer that scales features to a given range, typically [0, 1].
    This is similar to scikit-learn's MinMaxScaler.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.data_min_: dict[str, float] = {}
        self.data_max_: dict[str, float] = {}
        self.columns: list[str] = []

        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of feature_range must be smaller than maximum.")
        self.feature_range = feature_range

    def fit(self, df: pl.DataFrame, columns: list[str]) -> 'MinMaxNormalizer':
        """
        Computes the minimum and maximum for the specified columns.

        Args:
            df: The Polars DataFrame to fit on.
            columns: A list of column names to normalize.

        Returns:
            The fitted normalizer instance.
        """
        self.columns = columns
        if not self.columns:
            logging.warning("Warning: No columns provided to fit.")
            return self

        stats = df.select(
            [pl.col(c).min().alias(f"{c}_min") for c in self.columns]
            + [pl.col(c).max().alias(f"{c}_max") for c in self.columns]
        )
        for col in self.columns:
            self.data_min_[col] = stats.select(f"{col}_min").item()
            self.data_max_[col] = stats.select(f"{col}_max").item()

        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """
        Scales the specified columns in the DataFrame to the `feature_range`
        and returns the result as a NumPy array.

        Args:
            df: The DataFrame to transform.

        Returns:
            A NumPy array of the transformed data.
        """
        if not self.columns:
            raise RuntimeError(
                "'MinMaxNormalizer' must be fitted before transform.")

        feature_min, feature_max = self.feature_range
        feature_range_size = feature_max - feature_min

        transform_expressions = []
        for col in self.columns:
            data_min = self.data_min_[col]
            data_max = self.data_max_[col]
            data_range = data_max - data_min

            # Handle constant columns to avoid division by zero
            if data_range > 1e-9:
                expr = (
                    ((pl.col(col) - data_min) / data_range) *
                    feature_range_size + feature_min
                )
            else:
                # If the column is constant, map them to feature_min
                expr = pl.lit(feature_min, dtype=pl.Float64)

            transform_expressions.append(expr)

        return df.select(transform_expressions).to_numpy()

    def inverse_transform(self, data: np.ndarray) -> pl.DataFrame:
        """
        Applies the inverse scaling.

        Args:
            data: A NumPy array of scaled data, with columns in the
                same order as they were fitted.

        Returns:
            A Polars DataFrame of the data scaled back to its original representation.
        """
        if not self.columns:
            raise RuntimeError(
                "'MinMaxNormalizer' must be fitted before inverse transform.")

        if data.shape[1] != len(self.columns):
            raise ValueError(
                f"Input array has {data.shape[1]} columns, but normalizer was "
                f"fitted on {len(self.columns)} columns."
            )

        df = pl.DataFrame(data, schema=self.columns, strict=False)

        feature_min, feature_max = self.feature_range
        feature_range_size = feature_max - feature_min

        inverse_expressions = []
        for col in self.columns:
            data_min = self.data_min_[col]
            data_max = self.data_max_[col]
            data_range = data_max - data_min

            if data_range > 1e-9:
                expr = (
                    ((pl.col(col) - feature_min) /
                        feature_range_size) * data_range + data_min
                )
            else:
                # If the original column was constant, all inverse values are that constant
                expr = pl.lit(data_min, dtype=pl.Float64)

            inverse_expressions.append(expr.alias(col))

        return df.with_columns(inverse_expressions)

    def get_params(self) -> dict:
        """Returns the learned parameters."""
        return {
            "feature_range": self.feature_range,
            "data_min_": self.data_min_,
            "data_max_": self.data_max_,
            "columns": self.columns,
        }
