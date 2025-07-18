from .base import BaseNormalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import polars as pl

DEFAULT_MAX_FEATURES = 64

class BoWNormalizer():
    def __init__(self, stop_words="english", max_features=None):
        self.vectorizers: Dict[str, CountVectorizer] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.stop_words = stop_words
        self.max_features = max_features if max_features is not None else DEFAULT_MAX_FEATURES
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
        self.max_features = max_features if max_features is not None else DEFAULT_MAX_FEATURES
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