from typing import Literal
import numpy as np
from typing import Optional
import polars as pl
from ..container import GraphDataContainer
from .base import PipelineStep


class TemporalSplitter(PipelineStep):
    def __init__(self, train_ratio: float = 0.7, val_ratio: Optional[float] = None, split_col: str = "split") -> None:
        super().__init__()
        if val_ratio is None:
            val_ratio = 0.0

        if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
            raise ValueError("Invalid ratios provided.")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.split_col = split_col

    def _get_split_counts(self, data: GraphDataContainer, level: Literal["node", "edge"]) -> dict[str, int]:
        split_df = data.edges if level == "edge" else data.nodes
        if split_df is None:
            raise ValueError(f"Split data is 'None' ({level}). Did you run '{self.__class__.__name__}' ?")

        vc = split_df[self.split_col].value_counts()
        return dict(zip(vc[self.split_col].to_list(), vc["count"].to_list()))

    def validate(self, data: GraphDataContainer | None) -> None:
        if data is None:
            raise ValueError("Input data is None.")

        edge_timestamps = data.edge_timestamps
        node_timestamps = data.node_timestamps

        # Validate nodes if they exist and have timestamps
        if data.nodes is not None and node_timestamps is None:
            self.logger.warning(
                f"Node features are present but missing a timestamp column. Nodes will not be split chronologically."
            )
        elif node_timestamps is not None and not (
            node_timestamps.dtype.is_numeric() or node_timestamps.dtype.is_temporal()
        ):
            raise ValueError(
                f"'{data.metadata.n_time_col}' column in '{data.__class__.__name__}' is not numerical nor temporal."
            )

        if not (edge_timestamps.dtype.is_numeric() or edge_timestamps.dtype.is_temporal()):
            raise ValueError(
                f"'{data.metadata.e_time_col}' column in '{data.__class__.__name__}' is not numerical nor temporal."
            )

    def process(self, data: GraphDataContainer | None) -> GraphDataContainer:
        assert data is not None
        self.logger.info(f"Total edges: {len(data.edges)}")
        self.logger.info(
            f"Performing chronological split on edges and nodes: "
            f"Train={self.train_ratio:.2f}, Val={self.val_ratio:.2f}, Test={self.test_ratio:.2f}"
        )

        # Determine timestamp cutoff times from edges
        edge_timestamps_array = data.edge_timestamps.to_numpy()
        train_cutoff_time, val_cutoff_time = np.quantile(
            edge_timestamps_array,
            [(1 - self.val_ratio - self.test_ratio), (1 - self.test_ratio)],
        )
        self.logger.info(f"Train cutoff time: {train_cutoff_time}, Validation cutoff time: {val_cutoff_time}")

        data.edges = data.edges.with_columns(
            pl.when(pl.col(data.metadata.e_time_col) <= train_cutoff_time)
            .then(pl.lit("train"))
            .when(pl.col(data.metadata.e_time_col) <= val_cutoff_time)
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias(self.split_col)
        )

        split_counts = self._get_split_counts(data, "edge")
        self.logger.info(f"Edge split counts: {split_counts}")

        # Split nodes
        # We use the same cutoff times derived from the edges (if possible)
        node_id_col = data.metadata.n_id_col
        node_timestamps = data.node_timestamps

        if node_timestamps is not None:
            assert data.nodes is not None
            self.logger.info(
                f"Applying same edge time cutoffs to split node features. Number of nodes: {len(data.nodes)}."
            )

            data.nodes = data.nodes.with_columns(
                pl.when(pl.col(data.metadata.n_time_col) <= train_cutoff_time)
                .then(pl.lit("train"))
                .when(pl.col(data.metadata.n_time_col) <= val_cutoff_time)
                .then(pl.lit("val"))
                .otherwise(pl.lit("test"))
                .alias(self.split_col)
            )
        else:
            # Split nodes based on first appearence in split edges
            self.logger.info("Determining the 'first appearance' split for all nodes in the graph.")

            src_col = data.metadata.e_src_col
            tgt_col = data.metadata.e_tgt_col

            train_edges = data.edges.filter(pl.col(self.split_col) == "train")
            val_edges = data.edges.filter(pl.col(self.split_col) == "val")

            # Identify unique nodes appearing in each edge split
            train_nodes = pl.concat([train_edges[src_col], train_edges[tgt_col]]).unique()
            val_nodes = pl.concat([val_edges[src_col], val_edges[tgt_col]]).unique()

            # Create DataFrames with the split label for each set of nodes
            train_nodes_df = train_nodes.to_frame(node_id_col).with_columns(pl.lit("train").alias(self.split_col))
            val_nodes_df = val_nodes.to_frame(node_id_col).with_columns(pl.lit("val").alias(self.split_col))

            # Map contains each node ID and the split of its first appearance.
            # Nodes that appear in both train and val will be labeled 'train'
            first_appearance_map = pl.concat([train_nodes_df, val_nodes_df]).group_by(node_id_col).first()

            if data.nodes is not None:
                # A nodes DataFrame already exists. We join our map to it
                self.logger.info(
                    f"An existing nodes DataFrame was found. Adding 'split' column. Number of nodes: {len(data.nodes)}."
                )

                data.nodes = data.nodes.join(first_appearance_map, on=node_id_col, how="left").with_columns(
                    pl.col(self.split_col).fill_null("test")
                )

            else:
                # No nodes DataFrame exists
                self.logger.info("No nodes DataFrame found. Creating one from all unique nodes in edges.")

                # Get all unique nodes from the entire graph
                all_nodes = pl.concat([data.edges[src_col], data.edges[tgt_col]]).unique()

                # Create the base nodes DataFrame with just the node_id column
                data.nodes = all_nodes.to_frame(node_id_col)

                # Join the first appearance map
                data.nodes = data.nodes.join(first_appearance_map, on=node_id_col, how="left").with_columns(
                    pl.col(self.split_col).fill_null("test")
                )

        split_counts = self._get_split_counts(data, "node")
        self.logger.info(f"Node split counts: {split_counts}")

        # Create cumulative 'active_nodes' sets
        self.logger.info("Generating cumulative active node sets from the nodes 'split' column.")

        train_node_ids = data.nodes.filter(pl.col("split") == "train")[node_id_col]
        val_node_ids = data.nodes.filter(pl.col("split") == "val")[node_id_col]
        test_node_ids = data.nodes.filter(pl.col("split") == "test")[node_id_col]

        train_set = set(train_node_ids.to_list())
        val_set = set(val_node_ids.to_list())
        test_set = set(test_node_ids.to_list())

        val_active_set = train_set.union(val_set)
        test_active_set = val_active_set.union(test_set)

        data.set_metadata("active_nodes", {"train": train_set, "val": val_active_set, "test": test_active_set})

        self.logger.info(
            f"Active nodes sets created: {len(train_set)} (train), "
            f"{len(val_active_set)} (train+val), "
            f"{len(test_active_set)} (test/total)."
        )

        return data

    def update_metadata(self, data: GraphDataContainer) -> None:
        data.set_metadata("is_split", True)
        data.set_metadata("split_col", self.split_col)
