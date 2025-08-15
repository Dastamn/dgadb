import polars as pl
from typing import Optional
from ..container import GraphDataContainer
from .base import PipelineStep


class GraphSanitizer(PipelineStep):
    def __init__(
        self,
        remove_self_loops: bool = True,
        remove_duplicates: bool = True,
        to_undirected: bool = True,
        reindex_nodes: bool = True,
        ignore_timestamps: bool = False,
        splits: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.remove_self_loops = remove_self_loops
        self.remove_duplicates = remove_duplicates
        self.to_undericted = to_undirected
        self.reindex_nodes = reindex_nodes
        self.ignore_timestamps = ignore_timestamps
        self.splits = splits

    def _remove_self_loops(self, edges_df: pl.DataFrame, src_col: str, tgt_col: str) -> pl.DataFrame:
        self.logger.info("Removing self-loops...")
        n_before = len(edges_df)
        edges_df_filtered = edges_df.filter(pl.col(src_col) != pl.col(tgt_col))
        n_after = len(edges_df_filtered)
        self.logger.info(f"Removed {n_before - n_after} self-loops.")
        return edges_df_filtered

    def _remove_duplicates(
        self, edges_df: pl.DataFrame, src_col: str, tgt_col: str, timestamp_col: Optional[str] = None
    ) -> pl.DataFrame:
        cols = [src_col, tgt_col]
        if timestamp_col is not None:
            cols.append(timestamp_col)
            self.logger.info("Removing duplicates...")
        else:
            self.logger.info("Removing duplicates, ignoring timestamps...")

        n_before = len(edges_df)
        edges_df_filtered = edges_df.unique(subset=cols, keep="first")
        n_after = len(edges_df_filtered)
        self.logger.info(f"Removed {n_before - n_after} duplicates.")
        return edges_df_filtered

    def _to_undirected(self, edges_df: pl.DataFrame, src_col: str, tgt_col: str) -> pl.DataFrame:
        self.logger.info("Converting to undirected graph...")
        return (
            edges_df.with_columns(
                pl.when(pl.col(src_col) > pl.col(tgt_col))
                .then(pl.col(tgt_col))
                .otherwise(pl.col(src_col))
                .alias("temp_src"),
                pl.when(pl.col(src_col) > pl.col(tgt_col))
                .then(pl.col(src_col))
                .otherwise(pl.col(tgt_col))
                .alias("temp_tgt"),
            )
            .drop([src_col, tgt_col])
            .rename({"temp_src": src_col, "temp_tgt": tgt_col})
        )

    def _reindex_nodes(self, data: GraphDataContainer) -> tuple[pl.DataFrame, Optional[pl.DataFrame]]:
        self.logger.info("Re-indexing nodes...")
        if self.splits:
            self.logger.warning("Re-indexing ignores train, test and validation splits.")

        src_col = data.metadata.e_src_col
        tgt_col = data.metadata.e_tgt_col

        all_nodes = pl.concat([data.edges[src_col], data.edges[tgt_col]]).unique().sort()

        node_mapping = all_nodes.to_frame("original_id").with_row_index("new_id")

        # Map edges
        remapped_edges_df = (
            data.edges.join(node_mapping, left_on=src_col, right_on="original_id")
            .rename({"new_id": f"new_{src_col}"})
            .join(node_mapping, left_on=tgt_col, right_on="original_id")
            .rename({"new_id": f"new_{tgt_col}"})
            .drop([src_col, tgt_col])
            .rename({f"new_{src_col}": src_col, f"new_{tgt_col}": tgt_col})
        )

        # Map nodes
        remapped_nodes_df = None
        if data.nodes is not None:
            node_id_col = data.metadata.n_id_col
            assert node_id_col is not None

            remapped_nodes_df = (
                data.nodes.join(node_mapping, left_on=node_id_col, right_on="original_id")
                .rename({"new_id": f"new_{node_id_col}"})
                .drop(node_id_col)
                .rename({f"new_{node_id_col}": node_id_col})
            )

        data.set_metadata("node_mapping", node_mapping)

        return remapped_edges_df, remapped_nodes_df

    def validate(self, data: GraphDataContainer | None) -> None:
        if data is None:
            raise ValueError("Input data is None.")

        if self.ignore_timestamps:
            # Raise ColumnNotFoundError if non-existent
            _ = data.edges.get_column(data.metadata.e_time_col)

        if self.splits:
            if not data.metadata.is_split:
                raise ValueError(f"'data' has not been split into train, test and validation sets.")

            for split in self.splits:
                expected_splits = {"train", "test", "val"}
                if split not in expected_splits:
                    raise ValueError(f"Unknown split '{split}, expected: {expected_splits}'")

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        assert data is not None

        excluded_edges_df = None
        edges_df = data.edges
        nodes_df = data.nodes

        if self.reindex_nodes:
            edges_df, nodes_df = self._reindex_nodes(data)

        if self.splits:
            assert data.metadata.split_col is not None

            excluded_edges_df = edges_df.filter(pl.col(data.metadata.split_col).is_in(self.splits).not_())
            edges_df = edges_df.filter(pl.col(data.metadata.split_col).is_in(self.splits))

        src_col = data.metadata.e_src_col
        tgt_col = data.metadata.e_tgt_col

        if self.remove_self_loops:
            edges_df = self._remove_self_loops(edges_df, src_col, tgt_col)

        if self.to_undericted:
            edges_df = self._to_undirected(edges_df, src_col, tgt_col)

        if self.remove_duplicates:
            edges_df = self._remove_duplicates(
                edges_df, src_col, tgt_col, None if self.ignore_timestamps else data.metadata.e_time_col
            )

        if excluded_edges_df is not None:
            edges_df = pl.concat([edges_df, excluded_edges_df], how="vertical")

        edges_df = edges_df.sort(data.metadata.e_time_col)

        data.edges = edges_df
        data.nodes = nodes_df

        return data

    def update_metadata(self, data: GraphDataContainer) -> None:
        data.set_metadata("is_sanitized", True)
