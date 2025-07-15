from typing import Protocol
from sklearn.preprocessing import StandardScaler
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Dict, runtime_checkable
from enum import Enum

@runtime_checkable
class BaseNormalizer(Protocol):

    def fit(self, df: pl.DataFrame, columns: list[str]) -> None: ...

    def transform(self, df: pl.DataFrame) -> pl.DataFrame: ...

    def get_params(self) -> dict: ...


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
            raise RuntimeError("StandardNormalizer must be fitted before transform.")
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
            raise RuntimeError("MinMaxNormalizer must be fitted before transform.")
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
                (self.data_max_[col] - self.data_min_[col]) + self.data_min_[col]
            ).alias(col)
            for col in self.columns
        ])
    
    def get_params(self) -> dict:
        return {
            "feature_range": self.feature_range,
            "data_min_": self.data_min_,
            "data_max_": self.data_max_,
        }


class BoWNormalizer():

    def __init__(self, stop_words="english", max_features=None):
        self.vectorizers: Dict[str, CountVectorizer] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.stop_words = stop_words
        self.max_features = max_features
        self.columns: List[str] = []

    def fit(self, df: pl.DataFrame, columns: list[str]) -> None:
        self.columns = columns
        for col in columns:
            vectorizer = CountVectorizer(stop_words=self.stop_words, max_features=self.max_features)
            texts = df[col].to_list()
            vectorizer.fit(texts)
            self.vectorizers[col] = vectorizer
            self.feature_names[col] = vectorizer.get_feature_names_out().tolist()

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.columns:
            raise RuntimeError("BoWNormalizer must be fitted before transform.")
        
        df_out = df.clone()
        for col in self.columns:
            texts = df[col].to_list()
            vec = self.vectorizers[col]
            X = vec.transform(texts).toarray()
            new_cols = [
                pl.Series(f"{col}_bow_{i}", X[:, i])
                for i, _ in enumerate(self.feature_names[col])
            ]
            df_out = df_out.drop(col).with_columns(new_cols)
        return df_out

    def get_params(self) -> dict:
        return {
            "stop_words": self.stop_words,
            "max_features": self.max_features,
            "feature_names": self.feature_names
        }
    
class TfidfNormalizer():

    def __init__(self, stop_words="english", max_features=None):
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.stop_words = stop_words
        self.max_features = max_features
        self.columns: List[str] = []

    def fit(self, df: pl.DataFrame, columns: list[str]) -> None:
        self.columns = columns
        for col in columns:
            vectorizer = TfidfVectorizer(stop_words=self.stop_words, max_features=self.max_features)
            texts = df[col].to_list()
            vectorizer.fit(texts)
            self.vectorizers[col] = vectorizer
            self.feature_names[col] = vectorizer.get_feature_names_out().tolist()

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.columns:
            raise RuntimeError("TdidfNormalizer must be fitted before transform.")
        
        df_out = df.clone()
        for col in self.columns:
            texts = df[col].to_list()
            vec = self.vectorizers[col]
            X = vec.transform(texts).toarray()
            new_cols = [
                pl.Series(f"{col}_tfidf_{i}", X[:, i])
                for i, _ in enumerate(self.feature_names[col])
            ]
            df_out = df_out.drop(col).with_columns(new_cols)
        return df_out

    def get_params(self) -> dict:
        return {
            "stop_words": self.stop_words,
            "max_features": self.max_features,
            "feature_names": self.feature_names
        }



NORMALIZER_REGISTRY = {
    "standard": StandardNormalizer,
    "minmax": MinMaxNormalizer,
    "bow": BoWNormalizer,
    "tfidf": TfidfNormalizer,
    #word2vec??
}# TODO do this with an enum class instead

def get_normalizer(name: str, for_str:bool) -> BaseNormalizer:
    if for_str:
        return NORMALIZER_REGISTRY[name.lower()](max_features=64)
    else:
        return NORMALIZER_REGISTRY[name.lower()]()