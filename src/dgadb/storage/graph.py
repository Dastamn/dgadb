import torch
from typing import Optional, Dict, Any
import polars as pl

N_KEYS = {"n_feat", "n_type", "n_id", "n_label"}
E_KEYS = {"e_pairs", "e_weight", "e_feat", "e_type", "e_id", "e_label"}

class Graph:
    def __init__(
        self,
        nodes: Optional[Dict[str, torch.Tensor]] = None,
        edges: Optional[Dict[str, torch.Tensor]] = None,
        timestamps: Optional[Dict[str, pl.DataFrame]] = None
    ):
        self._nodes: Dict[str, torch.Tensor] = {}
        self._edges: Dict[str, torch.Tensor] = {}
        self.timestamps: Dict[str, pl.DataFrame] = timestamps

        if nodes:
            for k, v in nodes.items():
                self._validate_and_set(k, v, is_node=True)

        if edges:
            for k, v in edges.items():
                self._validate_and_set(k, v, is_node=False)

    def _validate_and_set(self, key: str, value: torch.Tensor, is_node: bool):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{'Node' if is_node else 'Edge'} key '{key}' must be a torch.Tensor")
        if is_node:
            if key not in N_KEYS:
                raise ValueError(f"Invalid node key: {key}")
            self._nodes[key] = value
        else:
            if key not in E_KEYS:
                raise ValueError(f"Invalid edge key: {key}")
            self._edges[key] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._nodes:
            return self._nodes[name]
        if name in self._edges:
            return self._edges[name]
        raise AttributeError(f"Graph object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name in {"_nodes", "_edges", "timestamps"}:
            super().__setattr__(name, value)
        elif name in N_KEYS:
            self._validate_and_set(name, value, is_node=True)
        elif name in E_KEYS:
            self._validate_and_set(name, value, is_node=False)
        else:
            raise AttributeError(f"Invalid attribute: {name}")

    def __repr__(self):
        node_keys = ", ".join(self._nodes.keys())
        edge_keys = ", ".join(self._edges.keys())
        timestamp_keys = ", ".join(self.timestamps.keys())
        return f"<Graph:\n  Nodes: [{node_keys}]\n  Edges: [{edge_keys}]\n  Timestamps: [{timestamp_keys}]>"

    def node_dict(self) -> Dict[str, torch.Tensor]:
        return self._nodes.copy()

    def edge_dict(self) -> Dict[str, torch.Tensor]:
        return self._edges.copy()

    def to(self, device: torch.device):
        for k in self._nodes:
            self._nodes[k] = self._nodes[k].to(device)
        for k in self._edges:
            self._edges[k] = self._edges[k].to(device)
        return self
    
    def node_timestamps(self) -> Optional[pl.DataFrame]:
        return self.timestamps.get("nodes", None)

    def edge_timestamps(self) -> Optional[pl.DataFrame]:
        return self.timestamps.get("edges", None)
    
    def set_node_timestamps(self, df: pl.DataFrame):
        self.timestamps["nodes"] = df

    def set_edge_timestamps(self, df: pl.DataFrame):
        self.timestamps["edges"] = df
    
    @property
    def num_nodes(self) -> int:
        if "n_id" in self._nodes:
            return self._nodes["n_id"].shape[0]
        return 0

    @property
    def num_edges(self) -> int:
        if "e_id" in self._edges:
            return self._edges["e_id"].shape[0]
        return 0

    @property
    def num_node_types(self) -> int:
        if "n_type" in self._nodes:
            return int(self._nodes["n_type"].unique().numel())
        return 1

    @property
    def num_edge_types(self) -> int:
        if "e_type" in self._edges:
            return int(self._edges["e_type"].unique().numel())
        return 1

    @property
    def node_types(self):
        if "n_type" in self._nodes:
            return self._nodes["n_type"].unique().cpu()
        return torch.tensor([0])

    @property
    def edge_types(self):
        if "e_type" in self._edges:
            return self._edges["e_type"].unique().cpu()
        return torch.tensor([0])

    @property
    def node_feature_dim(self) -> int:
        if "n_feat" in self._nodes:
            return self._nodes["n_feat"].shape[1]
        return 0

    @property
    def edge_feature_dim(self) -> int:
        if "e_feat" in self._edges:
            return self._edges["e_feat"].shape[1]
        return 0

# TODO add translators to turn them into pyg objects