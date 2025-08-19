import polars as pl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Optional, Literal
from .base import Normalizer

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

DEFAULT_MAX_FEATURES = 64
_TEXT_NORMALIZER_TYPES = ["bow", "tfidf"]


class TextNormalizer(Normalizer):
    def __init__(
        self, type: Literal["bow", "tfidf"] = "bow", stop_words: str = "english", max_features: Optional[int] = 64
    ) -> None:
        super().__init__()
        if type not in _TEXT_NORMALIZER_TYPES:
            raise ValueError(f"'TextNormalizer' type must be one of: {_TEXT_NORMALIZER_TYPES}")

        self.type = type
        self.stop_words = stop_words
        self.max_features = max_features

        self.columns: list[str] = []
        self.fitted_vectorizers_: dict[str, CountVectorizer | TfidfVectorizer] = {}

    def fit(self, df: pl.DataFrame, columns: list[str]) -> Self:
        if not columns:
            self.logger.warning("No columns provided to fit. Normalizer will do nothing.")
            return self

        self._validate_columns_exist(df, columns)
        self.columns = columns
        self.logger.info(f"Fitting TextNormalizer on columns: {self.columns}")

        for col in columns:
            if self.type == "bow":
                vectorizer = CountVectorizer(stop_words=self.stop_words, max_features=self.max_features)
            else:  # "tfidf"
                vectorizer = TfidfVectorizer(stop_words=self.stop_words, max_features=self.max_features)

            texts = df[col].drop_nulls().to_list()
            vectorizer.fit(texts)
            self.fitted_vectorizers_[col] = vectorizer

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted_vectorizers_:
            self.logger.warning("Normalizer has not been fitted.")
            return df

        cols_to_keep = [c for c in df.columns if c not in self.columns]
        transformed_df_list: list[pl.DataFrame] = [df.select(cols_to_keep)]

        self._validate_columns_exist(df, self.columns)
        for col in self.columns:
            if col not in self.fitted_vectorizers_:
                self.logger.warning(f"No fitted vectorizer found for column '{col}' Skipping.")
                transformed_df_list.append(df.select(col))
                continue

            self.logger.info(f"Transforming column '{col}' into '{self.type}' features...")

            vectorizer = self.fitted_vectorizers_[col]
            texts_to_transform = df[col].fill_null("").to_list()

            transformed_array = vectorizer.transform(texts_to_transform).toarray()

            feature_names = vectorizer.get_feature_names_out()
            new_col_names = [f"{col}_{self.type}_{name}" for name in feature_names]

            features_df = pl.DataFrame(transformed_array, schema=new_col_names)
            transformed_df_list.append(features_df)

        return pl.concat(transformed_df_list, how="horizontal")

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError(
            "Inverse transform is not supported for BoW or TF-IDF, as the original text cannot be perfectly recovered."
        )


# class SimpleTextNormalizer:
#     """
#     A normalizer that converts text columns into numerical vectors using either
#     Bag of Words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF).
#     """

#     def __init__(
#         self,
#         normalizer_type: Literal["bow", "tfidf"] = "bow",
#         stop_words: str = "english",
#         max_features: Optional[int] = None
#     ) -> None:
#         """
#         Initializes the text normalizer.

#         Args:
#             normalizer: The vectorization strategy to use ('bow' or 'tfidf').
#             stop_words: The language for stop words removal (e.g., "english").
#             max_features: The maximum number of features (words) to keep in the
#                           vocabulary, ranked by term frequency.
#         """
#         if normalizer_type not in ["bow", "tfidf"]:
#             raise ValueError("normalizer must be either 'bow' or 'tfidf'")

#         self.normalizer_type = normalizer_type
#         self.stop_words = stop_words
#         self.max_features = max_features

#         self.vectorizers: Dict[str, CountVectorizer | TfidfVectorizer] = {}
#         self.columns: List[str] = []

#     def fit(self, df: pl.DataFrame, columns: List[str]) -> 'SimpleTextNormalizer':
#         """
#         Fits a vectorizer (BoW or TF-IDF) to each specified text column.
#         """
#         self.columns = columns
#         if not self.columns:
#             print("Warning: No columns provided to fit.")
#             return self

#         for col in self.columns:
#             if self.normalizer_type == "bow":
#                 vectorizer = CountVectorizer(
#                     stop_words=self.stop_words, max_features=self.max_features
#                 )
#             else:  # "tfidf"
#                 vectorizer = TfidfVectorizer(
#                     stop_words=self.stop_words, max_features=self.max_features
#                 )

#             vectorizer.fit(df[col].to_list())
#             self.vectorizers[col] = vectorizer

#         return self

#     def transform(self, df: pl.DataFrame) -> np.ndarray:
#         """
#         Transforms text columns into numerical vectors and concatenates them
#         into a single NumPy array.
#         """
#         if not self.columns:
#             raise RuntimeError(
#                 "'SimpleTextNormalizer' must be fitted before transform.")

#         transformed_cols = []

#         for col in self.columns:
#             vectorizer = self.vectorizers[col]
#             texts = df[col].to_list()
#             vectorized = vectorizer.transform(texts).toarray()
#             transformed_cols.append(vectorized)

#         if not transformed_cols:
#             return np.array([[] for _ in range(len(df))])

#         return np.concatenate(transformed_cols, axis=1)

#     def inverse_transform(self, data: np.ndarray) -> None:
#         """Inverse transforming text vectors is not possible."""
#         raise NotImplementedError(
#             "Inverse transform is not supported for BoW or TF-IDF, as the "
#             "original text cannot be perfectly recovered."
#         )

#     def get_params(self) -> dict:
#         """Returns the learned parameters."""
#         feature_names = {
#             col: vec.get_feature_names_out().tolist()
#             for col, vec in self.vectorizers.items()
#         }
#         return {
#             "normalizer_type": self.normalizer_type,
#             "stop_words": self.stop_words,
#             "max_features": self.max_features,
#             "columns": self.columns,
#             "feature_names": feature_names
#         }


# class BoWNormalizer():
#     def __init__(self, stop_words="english", max_features=None):
#         self.vectorizers: Dict[str, CountVectorizer] = {}
#         self.feature_names: Dict[str, List[str]] = {}
#         self.stop_words = stop_words
#         self.max_features = max_features if max_features is not None else DEFAULT_MAX_FEATURES
#         self.columns: List[str] = []

#     def fit(self, df: pl.DataFrame, columns: list[str]) -> None:
#         self.columns = columns
#         for col in columns:
#             vectorizer = CountVectorizer(
#                 stop_words=self.stop_words, max_features=self.max_features)
#             texts = df[col].to_list()
#             vectorizer.fit(texts)
#             self.vectorizers[col] = vectorizer
#             self.feature_names[col] = vectorizer.get_feature_names_out(
#             ).tolist()

#     def transform(self, df: pl.DataFrame) -> pl.DataFrame:
#         if not self.columns:
#             raise RuntimeError(
#                 "BoWNormalizer must be fitted before transform.")

#         df_out = df.clone()
#         for col in self.columns:
#             texts = df[col].to_list()
#             vec = self.vectorizers[col]
#             X = vec.transform(texts).toarray()
#             new_cols = [
#                 pl.Series(f"{col}_bow_{i}", X[:, i])
#                 for i, _ in enumerate(self.feature_names[col])
#             ]
#             df_out = df_out.drop(col).with_columns(new_cols)
#         return df_out

#     def get_params(self) -> dict:
#         return {
#             "stop_words": self.stop_words,
#             "max_features": self.max_features,
#             "feature_names": self.feature_names
#         }


# class TfidfNormalizer():

#     def __init__(self, stop_words="english", max_features=None):
#         self.vectorizers: Dict[str, TfidfVectorizer] = {}
#         self.feature_names: Dict[str, List[str]] = {}
#         self.stop_words = stop_words
#         self.max_features = max_features if max_features is not None else DEFAULT_MAX_FEATURES
#         self.columns: List[str] = []

#     def fit(self, df: pl.DataFrame, columns: list[str]) -> None:
#         self.columns = columns
#         for col in columns:
#             vectorizer = TfidfVectorizer(
#                 stop_words=self.stop_words, max_features=self.max_features)
#             texts = df[col].to_list()
#             vectorizer.fit(texts)
#             self.vectorizers[col] = vectorizer
#             self.feature_names[col] = vectorizer.get_feature_names_out(
#             ).tolist()

#     def transform(self, df: pl.DataFrame) -> pl.DataFrame:
#         if not self.columns:
#             raise RuntimeError(
#                 "TdidfNormalizer must be fitted before transform.")

#         df_out = df.clone()
#         for col in self.columns:
#             texts = df[col].to_list()
#             vec = self.vectorizers[col]
#             X = vec.transform(texts).toarray()
#             new_cols = [
#                 pl.Series(f"{col}_tfidf_{i}", X[:, i])
#                 for i, _ in enumerate(self.feature_names[col])
#             ]
#             df_out = df_out.drop(col).with_columns(new_cols)
#         return df_out

#     def get_params(self) -> dict:
#         return {
#             "stop_words": self.stop_words,
#             "max_features": self.max_features,
#             "feature_names": self.feature_names
#         }
