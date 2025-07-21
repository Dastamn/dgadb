import polars as pl
from typing import Dict
from .base import BaseNormalizer
from .numerical import StandardNormalizer, MinMaxNormalizer
from .textual import BoWNormalizer, TfidfNormalizer
import logging
logger = logging.getLogger(__name__)

NORMALIZER_REGISTRY = {
    "standard": StandardNormalizer,
    "minmax": MinMaxNormalizer,
    "bow": BoWNormalizer,
    "tfidf": TfidfNormalizer,
    # word2vec??
}  # TODO do this with an enum class instead


def _get_normalizer(name: str, for_str: bool, max_features: int = None) -> BaseNormalizer:
    logger.debug(
        f"Instantiating normalizer: {name}, for_str={for_str}, max_features={max_features}")
    if for_str:
        return NORMALIZER_REGISTRY[name.lower()](max_features=max_features)
    else:
        return NORMALIZER_REGISTRY[name.lower()]()


def _apply_normalizers(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    train_df = df.filter(pl.col("split") == "train")
    for norm_type, columns in config.items():
        logger.info(
            f"Applying {norm_type} normalization to columns: {columns}")
        if norm_type in {"bow", "tfidf"}:
            for colconf in columns:
                if isinstance(colconf, dict):
                    col = colconf["name"]
                    max_features = colconf.get("max_features", None)
                else:
                    col = colconf
                    max_features = None
                if col not in df.columns:
                    logger.warning(
                        f"Column '{col}' not found in dataframe. Skipping normalization for this column.")
                    continue

                normalizer = _get_normalizer(
                    norm_type, for_str=True, max_features=max_features)
                logger.debug(
                    f"Fitting {norm_type} normalizer on column: {col}")
                normalizer.fit(train_df, [col])
                logger.debug(
                    f"Transforming all data using {norm_type} normalizer on column: {col}")
                df = normalizer.transform(df)

        else:
            valid_columns = [col for col in columns if col in df.columns]
            missing_columns = set(columns) - set(valid_columns)
            for missing in missing_columns:
                logger.warning(
                    f"Column '{missing}' not found in dataframe. Skipping normalization for this column.")
            if not valid_columns:
                logger.info(
                    f"No valid columns found for normalizer '{norm_type}'. Skipping.")
                continue
            normalizer = _get_normalizer(norm_type, for_str=False)
            logger.debug(
                f"Fitting {norm_type} normalizer on columns: {valid_columns}")
            normalizer.fit(df, valid_columns)
            logger.debug(
                f"Transforming all data using {norm_type} normalizer on columns: {valid_columns}")
            df = normalizer.transform(df)
    return df


def normalize_dataframes(data: Dict[str, pl.DataFrame], config: dict) -> Dict[str, pl.DataFrame]:
    if "nodes" in data and config.get("nodes", {}).get("normalizers"):
        logger.info("Normalizing node features...")
        data["nodes"] = _apply_normalizers(
            data["nodes"], config["nodes"]["normalizers"])

    if "edges" in data and config.get("edges", {}).get("normalizers"):
        logger.info("Normalizing edge features...")
        data["edges"] = _apply_normalizers(
            data["edges"], config["edges"]["normalizers"])

    return data
