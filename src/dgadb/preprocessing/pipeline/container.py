import polars as pl
from typing import Optional, Literal, Any
from dataclasses import dataclass, field


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
