from typing import Any
from typing import Any
import numpy as np
import polars as pl
from .base import BaseNormalizer
from .numerical import StandardNormalizer, MinMaxNormalizer
from .textual import SimpleTextNormalizer
import logging
logger = logging.getLogger(__name__)


NORMALIZER_REGISTRY = {
    "std": StandardNormalizer,
    "minmax": MinMaxNormalizer,
    "bow": SimpleTextNormalizer,
    "tfidf": SimpleTextNormalizer,
}


def _instantiate_normalizer(name: str, **kwargs: Any) -> BaseNormalizer:
    """
    Factory function to instantiate the correct normalizer with its specific arguments.
    """
    normalizer_class = NORMALIZER_REGISTRY.get(name.lower())
    if not normalizer_class:
        raise ValueError(f"Unknown normalizer type: {name}")

    if name.lower() in ["bow", "tfidf"]:
        kwargs["normalizer_type"] = name.lower()

    return normalizer_class(**kwargs)


def _apply_normalizers(
        df: pl.DataFrame,
        train_df: pl.DataFrame,
        norm_config: dict[str, dict]
) -> np.ndarray | None:
    """
    Applies a series of normalizers to a DataFrame and returns a single,
    concatenated NumPy feature matrix.

    Args:
        df: The full DataFrame to transform (containing all splits).
        train_df: The training subset of the DataFrame, used for fitting.
        norm_config: The normalizer configuration dictionary for this DataFrame.

    Returns:
        The final, concatenated feature matrix.
    """
    features = []
    for norm_type, column_config_array in norm_config.items():

        for column_config in column_config_array:
            target_cols = column_config.get("columns", [])
            norm_kwargs = column_config.get("params", {})

            # Check if config columns are present in df
            valid_columns = [col for col in target_cols if col in df.columns]
            if not valid_columns:
                continue

            normalizer = _instantiate_normalizer(norm_type, **norm_kwargs)
            normalizer.fit(train_df, valid_columns)
            feature_matrix = normalizer.transform(df)
            features.append(feature_matrix)

    return np.concatenate(features, axis=1) if features else None


def get_normalized_feature_matrices(
        nodes: pl.DataFrame | None,
        edges: pl.DataFrame,
        config: dict[str, dict]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Orchestrates the entire feature normalization for graph data.

    This function serves as the main entry point for feature processing. It
    takes the node and edge DataFrames, which are expected to have a boolean
    `train_mask` column, and a configuration dictionary that specifies how
    to normalize features.

    Args:
        nodes: An optional Polars DataFrame containing node features and a
               boolean `train_mask` column. Can be None if there are no node features.
        edges: A Polars DataFrame containing edge features and a boolean
               `train_mask` column.
        config: The full configuration dictionary, typically loaded from a YAML
                file, detailing the normalizers for nodes and edges.

    Returns:
        A tuple containing:
            - node_feature_matrix (np.ndarray | None): The final feature matrix for nodes, or None if no features were normalized.
            - edge_feature_matrix (np.ndarray | None): The final feature matrix for edges, or None if no features were normalized.
    """
    node_feature_matrix, edge_feature_matrix = None, None

    if nodes is not None and config.get("nodes", {}).get("normalizers"):
        train_nodes = nodes.filter(pl.col("train_mask"))
        node_feature_matrix = _apply_normalizers(
            nodes,
            train_nodes,
            config["nodes"]["normalizers"]
        )

    if config.get("edges", {}).get("normalizers"):
        train_edges = edges.filter(pl.col("train_mask"))
        edge_feature_matrix = _apply_normalizers(
            edges,
            train_edges,
            config["edges"]["normalizers"]
        )

    return node_feature_matrix, edge_feature_matrix


# NORMALIZER_REGISTRY = {
#     "standard": StandardNormalizer,
#     "minmax": MinMaxNormalizer,
#     "bow": BoWNormalizer,
#     "tfidf": TfidfNormalizer,
#     # word2vec??
# }  # TODO do this with an enum class instead


# def _get_normalizer(name: str, for_str: bool, max_features: int = None) -> BaseNormalizer:
#     logger.debug(
#         f"Instantiating normalizer: {name}, for_str={for_str}, max_features={max_features}")
#     if for_str:
#         return NORMALIZER_REGISTRY[name.lower()](max_features=max_features)
#     else:
#         return NORMALIZER_REGISTRY[name.lower()]()


# def _apply_normalizers(df: pl.DataFrame, config: dict) -> pl.DataFrame:
#     train_df = df.filter(pl.col("split") == "train")
#     for norm_type, columns in config.items():
#         logger.info(
#             f"Applying {norm_type} normalization to columns: {columns}")
#         if norm_type in {"bow", "tfidf"}:
#             for colconf in columns:
#                 if isinstance(colconf, dict):
#                     col = colconf["name"]
#                     max_features = colconf.get("max_features", None)
#                 else:
#                     col = colconf
#                     max_features = None
#                 if col not in df.columns:
#                     logger.warning(
#                         f"Column '{col}' not found in dataframe. Skipping normalization for this column.")
#                     continue

#                 normalizer = _get_normalizer(
#                     norm_type, for_str=True, max_features=max_features)
#                 logger.debug(
#                     f"Fitting {norm_type} normalizer on column: {col}")
#                 normalizer.fit(train_df, [col])
#                 logger.debug(
#                     f"Transforming all data using {norm_type} normalizer on column: {col}")
#                 df = normalizer.transform(df)

#         else:
#             valid_columns = [col for col in columns if col in df.columns]
#             missing_columns = set(columns) - set(valid_columns)
#             for missing in missing_columns:
#                 logger.warning(
#                     f"Column '{missing}' not found in dataframe. Skipping normalization for this column.")
#             if not valid_columns:
#                 logger.info(
#                     f"No valid columns found for normalizer '{norm_type}'. Skipping.")
#                 continue
#             normalizer = _get_normalizer(norm_type, for_str=False)
#             logger.debug(
#                 f"Fitting {norm_type} normalizer on columns: {valid_columns}")
#             normalizer.fit(df, valid_columns)
#             logger.debug(
#                 f"Transforming all data using {norm_type} normalizer on columns: {valid_columns}")
#             df = normalizer.transform(df)
#     return df


# def normalize_dataframes(data: Dict[str, pl.DataFrame], config: dict) -> Dict[str, pl.DataFrame]:
#     if "nodes" in data and config.get("nodes", {}).get("normalizers"):
#         logger.info("Normalizing node features...")
#         data["nodes"] = _apply_normalizers(
#             data["nodes"], config["nodes"]["normalizers"])

#     if "edges" in data and config.get("edges", {}).get("normalizers"):
#         logger.info("Normalizing edge features...")
#         data["edges"] = _apply_normalizers(
#             data["edges"], config["edges"]["normalizers"])

#     return data
