import polars as pl
from typing import Optional, Literal, Any, NamedTuple
from dataclasses import dataclass, field


class _GraphActiveNodes(NamedTuple):
    train: set[int]
    val: set[int]
    test: set[int]


@dataclass
class SplitData:
    edges: pl.DataFrame
    nodes: Optional[pl.DataFrame]


@dataclass
class GraphMetadata:
    # Core columns
    e_src_col: str = "src"
    e_tgt_col: str = "tgt"
    e_time_col: str = "timestamp"
    e_id_col: str = "edge_id"

    n_time_col: str = "timestamp"
    n_id_col: str = "node_id"

    f_id_col: str = "feature_id"
    f_val_col: str = "value"
    f_col_prefix: str = "feat"

    # State flags (set by pipeline steps)
    is_split: bool = False
    is_sanitized: bool = False
    is_feature_normalized: bool = False
    is_timestamp_normalized: bool = False

    # Artifacts from Pipeline steps
    split_col: Optional[str] = None  # from TemporalSplitter
    node_mapping: Optional["pl.DataFrame"] = None  # From GraphSanitizer
    active_nodes: Optional[dict[str, set[int]]] = None  # From TemporalSplitter

    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        return getattr(self, key, self.extra.get(key))

    def set(self, key: str, value: Any) -> None:
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def update(self, metadata: dict[str, Any]) -> None:
        self.__dict__.update(**metadata)

    def validate_columns(self, edge_df: pl.DataFrame, node_df: Optional[pl.DataFrame]) -> None:
        def _validate(col: str, expected_cols: list[str], source: Literal["edge", "node"]):
            if col not in expected_cols:
                raise ValueError(f"'{col}' not in {source} DataFrame, expected: {expected_cols}")

        for k, v in self.__dict__.items():
            if k.startswith("e_"):
                _ = [_validate(c, edge_df.columns, "edge") for c in edge_df.columns]
            elif k.startswith("n_") and node_df is not None:
                _ = [_validate(c, node_df.columns, "node") for c in node_df.columns]
                pass
            else:
                continue

    def keys(self) -> list[str]:
        return [*self.__dict__.keys(), *self.extra.keys()]


@dataclass
class GraphDataContainer:
    edges: pl.DataFrame
    nodes: Optional[pl.DataFrame]
    metadata: GraphMetadata = field(default_factory=GraphMetadata)

    def _get_split_data(self, split_name: Literal["train", "test", "val"]) -> SplitData:
        if not self.metadata.is_split:
            raise AttributeError("Data is not split.")

        split_col = self.metadata.split_col
        if not split_col:
            raise ValueError("'split_col' not found in metadata.")

        split_edges = self.edges.filter(pl.col(split_col) == split_name)
        split_nodes = None

        if self.nodes is not None:
            # For static nodes, we return the full node set
            # For dynamic nodes, we filter by split
            if split_col in self.nodes.columns:
                split_nodes = self.nodes.filter(pl.col(split_col) == split_name)
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
    def active_nodes(self) -> _GraphActiveNodes:
        if not self.metadata.is_split:
            raise AttributeError("Data is not split.")

        active_nodes = self.metadata.active_nodes
        if active_nodes is None:
            raise ValueError("'active_nodes' not found in metadata after splitting.")

        return _GraphActiveNodes(**active_nodes)

    @property
    def edge_timestamps(self) -> pl.Series:
        return self.edges[self.metadata.e_time_col]

    @property
    def node_timestamps(self) -> Optional[pl.Series]:
        return (
            self.nodes[self.metadata.n_time_col]
            if self.nodes is not None and self.metadata.n_time_col in self.nodes.columns
            else None
        )

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata.set(key, value)

    def describe(self) -> None:
        def _pretty_print(key: str, value: Any):
            if isinstance(value, (list, set, dict)):
                print(f"  - {key}: <{type(value).__name__} of length {len(value)}>")
            elif isinstance(value, pl.DataFrame):
                print(f"  - {key}: <Polars DataFrame of shape {value.shape}>")
            else:
                print(f"  - {key}: {value}")

        print("--- GraphDataContainer Summary ---")
        print(f"Edges DataFrame of shape {self.edges.shape}")
        if self.nodes is not None:
            print(f"Nodes DataFrame of shape {self.nodes.shape}")
        else:
            print("Nodes DataFrame: None")

        print("\nMetadata:")
        for key, value in self.metadata.__dict__.items():
            if key == "extra" or value is None:
                continue
            _pretty_print(key, value)

        if self.metadata.extra:
            print("\nExtra Metadata:")
            for key, value in self.metadata.extra.items():
                _pretty_print(key, value)

        print("---------------------------------")

    def __repr__(self) -> str:
        node_shape = self.nodes.shape if self.nodes is not None else "None"
        return (
            f"{self.__class__.__name__}("
            f"edges={self.edges.shape}, "
            f"nodes={node_shape}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )
