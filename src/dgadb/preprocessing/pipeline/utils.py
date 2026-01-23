from dgadb.preprocessing.pipeline import StructureNormalizer, Pipeline
from .container import GraphDataContainer
from pathlib import Path
import polars as pl
from dgadb.models.baseline.gnn import GNNAD
import logging

logging.basicConfig(level=logging.INFO)


def load_custom_dataset(data_dir: str, src_col: str = "src", tgt_col: str = "tgt", time_col: str = "t", label_col: str = "label") -> GraphDataContainer:
    data_path = Path(data_dir)

    df_train = pl.read_csv(data_path / "train.csv")
    df_test = pl.read_csv(data_path / "test.csv")

    train_cols = df_train.columns
    if label_col in train_cols:
        pass  # Already exists
    else:
        df_train = df_train.with_columns(
            pl.lit(0, dtype=pl.Int64).alias(label_col))

    df_train = df_train.with_columns(pl.lit("train").alias("split"))

    df_test = df_test.with_columns([
        pl.lit("test").alias("split")
    ])

    edges_df = pl.concat([df_train, df_test], how="diagonal")
    edges_df = edges_df.with_row_index("edge_id")

    container = GraphDataContainer(
        edges=edges_df,
        nodes=None,
        e_src_col=src_col,
        e_tgt_col=tgt_col,
        e_time_col=time_col,
        e_label_col=label_col
    )

    container.update_metadata({
        "is_split": True,
        "split_col": "split",
        "is_feature_normalized": False,
        "is_timestamp_normalized": False,
        "reindex_nodes": False
    })

    return container


if __name__ == "__main__":
    from dgadb.experiment.runner import ExperimentRunner

    cont = load_custom_dataset("anom_gen/as-topology_0.5_0.1")
    cont.describe()

    pipeline = Pipeline([StructureNormalizer("canonical")])
    cont = pipeline.run(cont)

    tg = cont.to_temporal_graph()
    tg.describe()

    runner = ExperimentRunner(GNNAD("GCN"), tg)
    runner.run(3, {"strategy": "window", "window_size": 2000,
               "include_cumulative": True}, evaluate=True)
