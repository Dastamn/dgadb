import polars as pl
from typing import Literal, Optional
from .base import PipelineStep
from ..container import GraphDataContainer


class StructureNormalizer(PipelineStep):
    """Pipeline step normalising graph structure (direction, dedup, reindex).

    Optionally removes self-loops, converts edges to ``"directed"``,
    ``"canonical"`` (``src < dst``) or ``"undirected"`` form, deduplicates
    by ``(src, tgt, t)``, and re-indexes node IDs to a contiguous
    ``0..N-1`` range.

    Args:
        directionality: Edge directionality target.
        reindex_nodes: Whether to re-index node IDs after dedup.
        remove_self_loops: Whether to drop ``src == tgt`` edges.
        remove_duplicates: Whether to drop duplicate edges in the directed case.
    """

    def __init__(
        self,
        directionality: Literal["directed",
                                "canonical", "undirected"],
        reindex_nodes: bool = True,
        remove_self_loops: bool = True,
        remove_duplicates: bool = True
    ) -> None:
        super().__init__()
        self.directionality = directionality
        self.reindex_nodes = reindex_nodes
        self.remove_self_loops = remove_self_loops
        self.remove_duplicates = remove_duplicates
        self.node_mapping: Optional[pl.DataFrame] = None

    def _to_canonical(self, df: pl.DataFrame, src_col: str, tgt_col: str) -> pl.DataFrame:
        return (
            df.with_columns(
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

    def _reindex_nodes(
        self,
        edge_df: pl.DataFrame,
        node_df: Optional[pl.DataFrame],
        src_col: str,
        tgt_col: str,
        node_id_col: Optional[str]
    ) -> tuple[pl.DataFrame, Optional[pl.DataFrame], pl.DataFrame]:
        all_nodes = pl.concat(
            [edge_df[src_col], edge_df[tgt_col]]).unique().sort()
        node_mapping = all_nodes.to_frame(
            "original_id").with_row_index("new_id")

        # Map edges
        remapped_edges_df = (
            edge_df.join(node_mapping, left_on=src_col,
                         right_on="original_id")
            .rename({"new_id": f"new_{src_col}"})
            .join(node_mapping, left_on=tgt_col, right_on="original_id")
            .rename({"new_id": f"new_{tgt_col}"})
            .drop([src_col, tgt_col])
            .rename({f"new_{src_col}": src_col, f"new_{tgt_col}": tgt_col})
        )

        # Map nodes
        remapped_nodes_df = None
        if node_df is not None:
            if node_id_col is None:
                raise ValueError('Node ID column not set, check DataLoader.')

            remapped_nodes_df = (
                node_df.join(node_mapping, left_on=node_id_col,
                             right_on="original_id")
                .rename({"new_id": f"new_{node_id_col}"})
                .drop(node_id_col)
                .rename({f"new_{node_id_col}": node_id_col})
            )

        return remapped_edges_df, remapped_nodes_df, node_mapping

    def validate(self, data: GraphDataContainer | None) -> None:
        """Verify that the input container is not ``None``.

        Args:
            data: The container from the previous step.

        Raises:
            ValueError: If ``data`` is ``None``.
        """
        if data is None:
            raise ValueError("Input data is None.")

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        """Apply directionality conversion, deduplication, and node re-indexing.

        Returns:
            The container with structurally normalised edges and, if
            ``reindex_nodes`` is ``True``, contiguous node IDs.
        """
        assert data is not None

        src_col = data.e_src_col
        tgt_col = data.e_tgt_col
        time_col = data.e_time_col

        if self.remove_self_loops:
            n_before = len(data.edges)
            data.edges = data.edges.filter(pl.col(src_col) != pl.col(tgt_col))
            self.logger.info(
                f"Removed {n_before - len(data.edges)} self-loops.")

        if self.directionality == "canonical":
            self.logger.info(
                "Converting to canonical representation (src < dst)...")
            canonical_edges = self._to_canonical(data.edges, src_col, tgt_col)
            n_before = len(canonical_edges)
            data.edges = canonical_edges.unique(
                subset=[src_col, tgt_col, time_col], keep="first")
            self.logger.info(
                f"Removed {n_before - len(data.edges)} duplicate edges after canonicalization.")

        elif self.directionality == "undirected":
            self.logger.info(
                "Converting to undirected graph (adding reverse edges)...")

            reverse_edges = data.edges.rename(
                {src_col: tgt_col, tgt_col: src_col})

            undirected_edges = pl.concat(
                [data.edges, reverse_edges], rechunk=True)

            n_before = len(undirected_edges)
            canonical_undirected = self._to_canonical(
                undirected_edges, src_col, tgt_col)
            data.edges = canonical_undirected.unique(
                subset=[src_col, tgt_col, time_col], keep="first")
            self.logger.info(
                f"Removed {n_before - len(data.edges)} duplicate/reverse edges after making undirected.")

        elif self.directionality == "directed" and self.remove_duplicates:
            n_before = len(data.edges)
            data.edges = data.edges.unique(
                subset=[src_col, tgt_col, time_col], keep="first")
            self.logger.info(
                f"Removed {n_before - len(data.edges)} duplicates.")

        else:
            raise NotImplementedError(
                f"Unknown directionality '{self.directionality}'")
        
        if self.reindex_nodes:
            self.logger.info(
                "Re-indexing nodes to a contiguous 0-to-N-1 range...")
            data.edges, data.nodes, self.node_mapping = self._reindex_nodes(
                data.edges, data.nodes, src_col, tgt_col, data.n_id_col)
            self.logger.info(
                f"Re-indexing complete. Found {len(self.node_mapping)} unique nodes.")

        return data

    def update_metadata(self, data: GraphDataContainer) -> None:
        """Record directionality settings and the node-ID mapping in metadata."""
        data.update_metadata({
            "directionality": self.directionality,
            "reindex_nodes": self.reindex_nodes,
            "remove_self_loops": self.remove_self_loops,
            "remove_duplicates": self.remove_duplicates,
            "node_mapping": self.node_mapping
        })
