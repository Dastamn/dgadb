from typing import List, runtime_checkable, Protocol
import polars as pl


@runtime_checkable
class BaseNormalizer(Protocol):

    def fit(self, df: pl.DataFrame, columns: list[str]) -> None: ...

    def transform(self, df: pl.DataFrame) -> pl.DataFrame: ...

    def get_params(self) -> dict: ...
