import logging
import polars as pl
from abc import ABC, abstractmethod

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class Normalizer(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_columns_exist(self, df: pl.DataFrame, columns: list[str]) -> None:
        df_columns_set = set(df.columns)
        for col in columns:
            if col not in df_columns_set:
                raise ValueError(
                    f"Column '{col}' not found in the DataFrame. Available columns are: {list(df_columns_set)}"
                )

    @abstractmethod
    def fit(self, df: pl.DataFrame, columns: list[str]) -> Self:
        """
        Computes the necessary statistics from the data.

        This method should only be called on the training set to prevent data
        leakage.

        Args:
            df: The DataFrame to fit on.
            columns: A list of column names to be processed by this normalizer.

        Returns:
            The fitted normalizer instance.
        """
        ...

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies the fitted transformation to the data.

        Args:
            df: The DataFrame to transform.

        Returns:
            A Polars Dataframe containing the only the transformed columns.
        """
        ...

    @abstractmethod
    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reverts the applied transformation, scaling the data back to its
        original representation.

        Note: Not all normalizers (e.g., text vectorizers) can support this
        operation, in which case they should raise a NotImplementedError.

        Args:
            data: A NumPy array of transformed data, with columns in the same
                  order as they were fitted.

        Returns:
            A Polars DataFrame containing the inverse-transformed data with
            appropriate column names.
        """
        ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if k != "logger")
        return f"{self.__class__.__name__}({params})"


# @runtime_checkable
# class BaseNormalizer(Protocol):
#     """
#     A protocol defining the standard interface for all normalizer classes.

#     Any class that implements these methods with the correct signatures
#     will be considered a valid normalizer, enabling structural subtyping.
#     """

#     def fit(self, df: pl.DataFrame, columns: List[str]) -> Self:
#         """
#         Computes the necessary statistics from the data.

#         This method should only be called on the training set to prevent data
#         leakage.

#         Args:
#             df: The DataFrame to fit on.
#             columns: A list of column names to be processed by this normalizer.

#         Returns:
#             The fitted normalizer instance.
#         """
#         ...

#     def transform(self, df: pl.DataFrame) -> np.ndarray:
#         """
#         Applies the fitted transformation to the data.

#         Args:
#             df: The DataFrame to transform.

#         Returns:
#             A NumPy array representing the transformed features.
#         """
#         ...

#     def inverse_transform(self, data: np.ndarray) -> pl.DataFrame:
#         """
#         Reverts the applied transformation, scaling the data back to its
#         original representation.

#         Note: Not all normalizers (e.g., text vectorizers) can support this
#         operation, in which case they should raise a NotImplementedError.

#         Args:
#             data: A NumPy array of transformed data, with columns in the same
#                   order as they were fitted.

#         Returns:
#             A Polars DataFrame containing the inverse-transformed data with
#             appropriate column names.
#         """
#         ...

#     def get_params(self) -> Dict:
#         """
#         Returns a dictionary containing the learned parameters of the normalizer.

#         Returns:
#             A dictionary of the fitted parameters.
#         """
#         ...
