from .base import BaseNormalizer
import polars as pl
from typing import Dict

# TODO fix typing


class StandardNormalizer():

    def __init__(self):
        self.mean_ = {}
        self.std_ = {}
        self.columns = []

    def fit(self, df: pl.DataFrame, columns: list[str]) -> None:
        self.columns = columns
        for col in columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            self.mean_[col] = col_mean
            self.std_[col] = col_std

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.columns:
            raise RuntimeError(
                "StandardNormalizer must be fitted before transform.")
        return df.with_columns([
            ((pl.col(col) - self.mean_[col]) / self.std_[col]).alias(col)
            for col in self.columns
        ])

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            ((pl.col(col) * self.std_[col]) + self.mean_[col]).alias(col)
            for col in self.columns
        ])

    def get_params(self) -> dict:
        return {"mean_": self.mean_, "std_": self.std_}


class MinMaxNormalizer():

    def __init__(self, feature_range=(0, 1)):
        self.min_ = {}
        self.max_ = {}
        self.data_min_ = {}
        self.data_max_ = {}
        self.columns = []
        self.feature_range = feature_range

    def fit(self, df: pl.DataFrame, columns: list[str]) -> None:
        self.columns = columns
        for col in columns:
            col_min = df[col].min()
            col_max = df[col].max()
            self.data_min_[col] = col_min
            self.data_max_[col] = col_max
            self.min_[col] = self.feature_range[0]
            self.max_[col] = self.feature_range[1]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.columns:
            raise RuntimeError(
                "MinMaxNormalizer must be fitted before transform.")
        return df.with_columns([
            (
                ((pl.col(col) - self.data_min_[col]) /
                 (self.data_max_[col] - self.data_min_[col])) *
                (self.max_[col] - self.min_[col]) + self.min_[col]
            ).alias(col)
            for col in self.columns
        ])

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            (
                ((pl.col(col) - self.min_[col]) /
                 (self.max_[col] - self.min_[col])) *
                (self.data_max_[col] - self.data_min_[col]) +
                self.data_min_[col]
            ).alias(col)
            for col in self.columns
        ])

    def get_params(self) -> dict:
        return {
            "feature_range": self.feature_range,
            "data_min_": self.data_min_,
            "data_max_": self.data_max_,
        }
