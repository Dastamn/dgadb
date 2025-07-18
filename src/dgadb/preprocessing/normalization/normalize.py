import polars as pl
from typing import Dict
from .base import BaseNormalizer
from .numerical import StandardNormalizer, MinMaxNormalizer
from .textual import BoWNormalizer, TfidfNormalizer

NORMALIZER_REGISTRY = {
    "standard": StandardNormalizer,
    "minmax": MinMaxNormalizer,
    "bow": BoWNormalizer,
    "tfidf": TfidfNormalizer,
    #word2vec??
}# TODO do this with an enum class instead

def _get_normalizer(name: str, for_str:bool, max_features: int = None ) -> BaseNormalizer:
    if for_str:
        return NORMALIZER_REGISTRY[name.lower()](max_features=max_features)
    else:
        return NORMALIZER_REGISTRY[name.lower()]()

def _apply_normalizers(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    train_df = df.filter(pl.col("split") == "train")
    for norm_type, columns in config.items():
        if norm_type in {"bow", "tfidf"}:
            for colconf in columns:
                if isinstance(colconf, dict):
                    col = colconf["name"]
                    max_features = colconf.get("max_features", None)
                else:
                    col = colconf
                    max_features = None
                if col not in df.columns:
                    print(f"Warning: column '{col}' not found. Skipping.")
                    continue
                
                normalizer = _get_normalizer(norm_type, for_str=True, max_features=max_features)
                normalizer.fit(train_df, [col])
                df = normalizer.transform(df)
                    
        else:
            valid_columns = [col for col in columns if col in df.columns]
            missing_columns = set(columns) - set(valid_columns)
            for missing in missing_columns:
                print(f"Warning: column '{missing}' not found. Skipping.")
            if not valid_columns:
                continue
            normalizer = _get_normalizer(norm_type, for_str=False)
            normalizer.fit(df, valid_columns)
            df = normalizer.transform(df)
    return df

def normalize_dataframes(data: Dict[str, pl.DataFrame], config: dict) -> Dict[str, pl.DataFrame]:
    if "nodes" in data and config.get("nodes", {}).get("normalizers"):
        data["nodes"] = _apply_normalizers(data["nodes"], config["nodes"]["normalizers"])

    if "edges" in data and config.get("edges", {}).get("normalizers"):
        data["edges"] = _apply_normalizers(data["edges"], config["edges"]["normalizers"])

    return data