import torch
from typing import Optional, Dict, Any
import polars as pl

N_KEYS = {
    "n_feat",
    "n_type",
    "n_id",
    "n_label",
    "n_snapshot_id",
    "n_timestamp",
    "n_train_mask",
    "n_val_mask",
    "n_test_mask",
}
E_KEYS = {
    "e_pairs",
    "e_weight",
    "e_feat",
    "e_type",
    "e_id",
    "e_label",
    "e_snapshot_id",
    "e_timestamp",
    "e_train_mask",
    "e_val_mask",
    "e_test_mask",
}


class Graph:
    def __init__(
        self,
        nodes: Optional[Dict[str, torch.Tensor]] = None,
        edges: Optional[Dict[str, torch.Tensor]] = None,
        window_size: Optional[int] = None,
    ):
        self._nodes: Dict[str, torch.Tensor] = {}
        self._edges: Dict[str, torch.Tensor] = {}
        self.window_size = window_size

        if nodes:
            for k, v in nodes.items():
                self._validate_and_set(k, v, is_node=True)

        if edges:
            for k, v in edges.items():
                self._validate_and_set(k, v, is_node=False)

        # assign global ids if not given
        node_ids = torch.unique(self._edges["e_pairs"])
        if "e_id" not in self._edges:
            num_edges = self._edges["e_pairs"].shape[1]
            self._edges["e_id"] = torch.arange(num_edges, dtype=torch.long)

        if "n_id" not in self._nodes and len(self._nodes) > 0:
            self._nodes["n_id"] = torch.arange(node_ids.size(0), dtype=torch.long)

        if "n_feat" not in self._nodes:
            self._nodes["n_feat"] = torch.eye(node_ids.size(0))

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
        if name in {"_nodes", "_edges", "window_size"}:
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
        return f"<Graph:\n  Nodes: [{node_keys}]\n  Edges: [{edge_keys}]\n>"

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

    def snapshots(self, all_nodes: bool = False):
        snapshot_id = 0
        num_snapshots = self.num_snapshots

        while snapshot_id < num_snapshots:
            e_mask = self._edges["e_snapshot_id"] <= snapshot_id
            e_idx = e_mask.nonzero(as_tuple=True)[0]
            edges_sub = {}
            for k, v in self._edges.items():
                if k == "e_pairs":
                    edges_sub[k] = v[:, e_idx]
                elif v.shape[0] == e_mask.shape[0]:
                    edges_sub[k] = v[e_idx]

            if "n_snapshot_id" in self._nodes:
                # if nodes have timestamps we filter these too
                n_mask = self._nodes["n_snapshot_id"] <= snapshot_id
                n_idx = n_mask.nonzero(as_tuple=True)[0]
                nodes_sub = {k: v[n_idx] for k, v in self._nodes.items() if v.shape[0] == n_mask.shape[0]}

            else:
                if all_nodes:
                    # include all nodes
                    nodes_sub = self._nodes
                else:
                    # just include nodes present in current edges
                    edge_node_ids = torch.unique(edges_sub["e_pairs"])
                    id_map = {id.item(): i for i, id in enumerate(edge_node_ids)}
                    nodes_sub = {}
                    mask = torch.tensor([x.item() in id_map for x in self._nodes["n_id"]])
                    n_idx = mask.nonzero(as_tuple=True)[0]
                    for k, v in self._nodes.items():
                        if v.shape[0] == mask.shape[0]:
                            nodes_sub[k] = v[n_idx]

            yield Graph(nodes=nodes_sub, edges=edges_sub)
            snapshot_id += 1

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
    def num_snapshots(self) -> int:
        if "n_snapshot_id" in self._nodes or "e_snapshot_id" in self._edges:
            if "n_snapshot_id" in self._nodes:
                n_snapshots = torch.max(self._nodes["n_snapshot_id"]).item()
            else:
                n_snapshots = 0
            if "e_snapshot_id" in self._edges:
                e_snapshots = torch.max(self._edges["e_snapshot_id"]).item()
            return max(n_snapshots, e_snapshots)
        return 0

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


# TODO make it so that different edge/node types can have differet feature dims
