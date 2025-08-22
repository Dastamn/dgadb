import logging
import torch
import polars as pl
from typing import Optional, Literal, Any
from dataclasses import dataclass, field
from src.dgadb.storage import TemporalGraph

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    edges: pl.DataFrame
    nodes: Optional[pl.DataFrame]


@dataclass
class GraphDataContainer:
    edges: pl.DataFrame
    nodes: Optional[pl.DataFrame]

    e_src_col: str
    e_tgt_col: str
    e_time_col: str
    e_id_col: str

    n_time_col: str
    n_id_col: str

    feat_id_col: str
    feat_val_col: str
    feat_col_prefix: str

    metadata: dict = field(default_factory=dict)

    def _get_split_data(self, split_name: Literal["train", "test", "val"]) -> SplitData:
        if not self.is_split:
            raise AttributeError("Data is not split.")

        split_col = self.split_col
        if not split_col:
            raise ValueError("'split_col' not found in metadata.")

        split_edges = self.edges.filter(pl.col(split_col) == split_name)
        split_nodes = None

        if self.nodes is not None:
            # For static nodes, we return the full node set
            # For dynamic nodes, we filter by split
            if split_col in self.nodes.columns:
                split_nodes = self.nodes.filter(
                    pl.col(split_col) == split_name)
            else:
                split_nodes = self.nodes

        return SplitData(edges=split_edges, nodes=split_nodes)

    @property
    def train_data(self) -> SplitData:
        return self._get_split_data("train")

    @property
    def val_data(self) -> SplitData:
        return self._get_split_data("val")

    @property
    def test_data(self) -> SplitData:
        return self._get_split_data("test")

    @property
    def edge_timestamps(self) -> pl.Series:
        return self.edges[self.e_time_col]

    @property
    def node_timestamps(self) -> Optional[pl.Series]:
        return (
            self.nodes[self.n_time_col]
            if self.nodes is not None and self.n_time_col in self.nodes.columns
            else None
        )

    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    def describe(self) -> None:
        print("--- GraphDataContainer Summary ---")
        print(f"Edges DataFrame of shape {self.edges.shape}")
        if self.nodes is not None:
            print(f"Nodes DataFrame of shape {self.nodes.shape}")
        else:
            print("Nodes DataFrame: None")

        print("\nMetadata:")
        for key, value in self.metadata.items():
            if isinstance(value, (list, set, dict)):
                print(
                    f"  - {key}: <{type(value).__name__} of length {len(value)}>")
            elif isinstance(value, pl.DataFrame):
                print(f"  - {key}: <Polars DataFrame of shape {value.shape}>")
            else:
                print(f"  - {key}: {value}")

        print("---------------------------------")

    def to_temporal_graph(self) -> TemporalGraph:
        if not self.is_split or not self.split_col:
            raise RuntimeError(
                "Data has not been split. Cannot create train/val/test masks.")

        if not self.reindex_nodes:
            logger.warning(
                "Nodes have not been explicitly reindexed. Proceeding with original node IDs.")

        if not self.is_feature_normalized:
            logger.warning(
                "Features have not been normalized. Proceeding with raw features.")

        if not self.is_timestamp_normalized:
            logger.warning(
                "Timestamps have not been normalized. Proceeding with raw timestamps.")

        # Sort by time and src node (for event snapshotting)
        edges_df = self.edges.sort([self.e_time_col, self.e_src_col])

        src = torch.tensor(
            edges_df[self.e_src_col].to_numpy(), dtype=torch.long)
        tgt = torch.tensor(
            edges_df[self.e_tgt_col].to_numpy(), dtype=torch.long)
        t = torch.tensor(
            edges_df[self.e_time_col].to_numpy(), dtype=torch.long)

        num_nodes = max(int(src.max()), int(tgt.max())) + 1

        # Handle edge features
        edge_feat_cols = [
            col for col in self.edges.columns if col.startswith(self.feat_col_prefix)]
        if edge_feat_cols:
            msg = torch.tensor(
                edges_df[edge_feat_cols].to_numpy(), dtype=torch.float32)
        else:
            msg = torch.empty((len(edges_df), 0), dtype=torch.float32)
            logger.info(
                "No edge features selected. 'msg' tensor will be empty.")

        # Handle node attributes
        node_attr = None
        if self.nodes is not None:
            node_feat_cols = [col for col in self.nodes.columns
                              if col.startswith(self.feat_col_prefix)]
            if node_feat_cols:
                # Ensure nodes are sorted by their final ID
                sorted_nodes = self.nodes.sort(self.n_id_col)
                # Sanity check for contiguity
                if not sorted_nodes[self.n_id_col].equals(pl.arange(0, len(sorted_nodes), eager=True)):
                    raise RuntimeError(
                        "Node IDs are not contiguous, nodes need to be re-indexed. Did you run 'StructureNormalizer' ?")

                node_attr = torch.tensor(
                    sorted_nodes[node_feat_cols].to_numpy(), dtype=torch.float32)

        if node_attr is None:
            logger.info("No node features found, 'node_attr' is None.")

        # Create masks
        train_mask = torch.from_numpy(
            (edges_df[self.split_col] == 'train').to_numpy())
        val_mask = torch.from_numpy(
            (edges_df[self.split_col] == 'val').to_numpy())
        test_mask = torch.from_numpy(
            (edges_df[self.split_col] == 'test').to_numpy())

        # Labels default to all-zeros
        # TODO @Dastamn: Handle original labels
        edge_labels = torch.zeros(src.numel(), dtype=torch.long)
        node_labels = torch.zeros(num_nodes, dtype=torch.long)

        # TODO @Dastamn: Handle weights

        excluded_metadata = {"is_split", "split_col", "node_mapping"}
        metadata = {k: v for k, v in self.metadata.items()
                    if k not in excluded_metadata and v is not None}

        return TemporalGraph(
            src=src,
            tgt=tgt,
            t=t,
            msg=msg,
            edge_labels=edge_labels,
            node_attr=node_attr,
            node_labels=node_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            metadata=metadata
        )

    def __getattr__(self, name):
        metadata = object.__getattribute__(self, "metadata")
        return metadata.get(name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dataclass_fields__ or name == "metadata":
            super().__setattr__(name, value)
        else:
            metadata = object.__getattribute__(self, "metadata")
            metadata[name] = value

    def __repr__(self) -> str:
        node_shape = self.nodes.shape if self.nodes is not None else "None"
        return (
            f"{self.__class__.__name__}("
            f"edges={self.edges.shape}, "
            f"nodes={node_shape}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )
