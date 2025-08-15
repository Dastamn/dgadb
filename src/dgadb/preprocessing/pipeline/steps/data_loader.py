from typing import Optional
import polars as pl
import os
from ..container import GraphDataContainer, GraphMetadata
from .base import PipelineStep


class DataLoader(PipelineStep):
    # TODO @Dastamn: update to data_path="data/structured"
    def __init__(
        self,
        dataset_dir: str,
        **metadata: Optional[dict],
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.metadata = GraphMetadata()
        if metadata is not None:
            self.metadata.update(metadata)

    def _pivot_if_needed(self, df: pl.DataFrame, id_col: str, file_suffix: str) -> pl.DataFrame:
        if {id_col, self.metadata.f_id_col, self.metadata.f_val_col}.issubset(df.columns):
            self.logger.debug(f"Pivoting features for {id_col} from file with suffix '{file_suffix}'")
            pivoted_df = df.pivot(index=id_col, on=self.metadata.f_id_col, values=self.metadata.f_val_col)

            feature_cols = [col for col in pivoted_df.columns if col != id_col]
            rename_mapping = {col: f"{self.metadata.f_col_prefix}_{col}_{file_suffix}" for col in feature_cols}

            return pivoted_df.rename(rename_mapping)

        return df

    def validate(self, data: GraphDataContainer | None) -> None:
        pass

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        # Load and join edge files
        edge_file = os.path.join(self.dataset_dir, "edges.parquet")
        if not os.path.exists(edge_file):
            error = FileNotFoundError(f"Required edge file not found: {edge_file}")
            self.logger.error(error)
            raise error

        self.logger.info(f"Loading required edge file: {edge_file}")
        edge_df = pl.read_parquet(edge_file)
        self.logger.debug(f"Loaded base edges shape: {edge_df.shape}")

        edge_feature_files = [
            "edge_types",
            "edge_features_cat",
            "edge_features_num",
            "edge_features_str",
            "edge_labels",
        ]

        for f_name in edge_feature_files:
            f_path = os.path.join(self.dataset_dir, f"{f_name}.parquet")
            if os.path.exists(f_path):
                self.logger.info(f"Loading optional edge file: {f_path}")
                df_extra = pl.read_parquet(f_path)
                file_suffix = f_name.split("_")[-1]  # cat/num/str/labels/types
                df_extra_pivoted = self._pivot_if_needed(df_extra, self.metadata.e_id_col, file_suffix)
                edge_df = edge_df.join(df_extra_pivoted, on=self.metadata.e_id_col, how="left")

        self.logger.debug(f"Edges loaded with columns: {edge_df.columns}")

        # Load and join node files
        node_files = [
            "node_timestamps",
            "node_types",
            "node_features_cat",
            "node_features_num",
            "node_features_str",
            "node_labels",
        ]

        loaded_node_dfs: list[pl.DataFrame] = []
        for f_name in node_files:
            f_path = os.path.join(self.dataset_dir, f"{f_name}.parquet")
            if os.path.exists(f_path):
                self.logger.info(f"Loading optional node file: {f_path}")
                df = pl.read_parquet(f_path)
                file_suffix = f_name.split("_")[-1]
                df_pivoted = self._pivot_if_needed(df, self.metadata.n_id_col, file_suffix)
                loaded_node_dfs.append(df_pivoted)

        node_df = None
        if loaded_node_dfs:
            self.logger.debug(f"Joining {len(loaded_node_dfs)} node DataFrame(s) on '{self.metadata.n_id_col}'")
            node_df = loaded_node_dfs[0]
            for i in range(1, len(loaded_node_dfs)):
                node_df = node_df.join(loaded_node_dfs[i], on=self.metadata.n_id_col, how="left")

            self.logger.debug(f"Nodes loaded with columns: {node_df.columns}")

        self.metadata.validate_columns(edge_df, node_df)

        return GraphDataContainer(edge_df, node_df, self.metadata)

    def update_metadata(self, data: GraphDataContainer) -> None:
        pass
