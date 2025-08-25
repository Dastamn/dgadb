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
    
    def generate_snapshots(self, snapshot_size: int, temporal_snapshots: bool = False):
        """
        Assigns snapshot IDs to edges based on either temporal or structural slicing.

        Parameters:
        snapshot_size : int
            The size or duration of each snapshot. For temporal slicing this represents
            the time window. For structural slicing this is the number of edges per snapshot.
        
        temporal_snapshots : bool, optional (default=False)
            If True, snapshots are generated based on temporal intervals using the 'e_timestamp' field.
            If False, snapshots are generated structurally by grouping a fixed number of edges.

        Notes:
        - This method modifies the graph in-place by adding or updating the 'e_snapshot_id' field in the edge dictionary.
        - If 'e_val_mask' is present, it ensures that validation and test snapshots do not overlap in time with training.
        - When `temporal_snapshots` is True:
            - Edges are assigned to snapshots based on the difference between their timestamp and the minimum timestamp.
            - Snapshots for validation and test sets are adjusted to not overlap with training/validation snapshots.
        - When `temporal_snapshots` is False:
            - Edges are split into chunks of size `snapshot_size`, separately for train, val, and test sets.
            - Snapshot IDs are assigned to each chunk.
        """
        e_ts = self._edges["e_timestamp"]
        device = e_ts.device
        e_snapshot_id = torch.empty_like(e_ts, dtype=torch.long)

        train_mask = self._edges["e_train_mask"]
        test_mask = self._edges["e_test_mask"]
        val_mask = self._edges["e_val_mask"] if "e_val_mask" in self._edges else None

        if temporal_snapshots:
            min_time = e_ts.min()

            def _assign_temporal(mask: torch.Tensor):
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    return idx, torch.empty(0, dtype=torch.long, device=device)
                snaps = ((e_ts[idx] - min_time) // snapshot_size).long()
                return idx, snaps

            train_idx, train_snaps = _assign_temporal(train_mask)
            if train_idx.numel() > 0:
                e_snapshot_id[train_idx] = train_snaps
                max_snap_train = int(train_snaps.max().item())
            else:
                max_snap_train = None

            if val_mask is not None:
                val_idx, val_snaps = _assign_temporal(val_mask)
                if val_idx.numel() > 0:
                    if max_snap_train is not None and int(val_snaps.min().item()) == max_snap_train:
                        val_snaps = val_snaps + 1
                    e_snapshot_id[val_idx] = val_snaps
                    max_snap_val = int(val_snaps.max().item())
                else:
                    max_snap_val = None
            else:
                val_idx = torch.empty(0, dtype=torch.long, device=device)
                max_snap_val = None

            test_idx, test_snaps = _assign_temporal(test_mask)
            if test_idx.numel() > 0:
                min_snap_test = int(test_snaps.min().item())
                if max_snap_val is None:
                    if max_snap_train is not None and max_snap_train == min_snap_test:
                        test_snaps = test_snaps + 1
                else:
                    if max_snap_val == min_snap_test:
                        test_snaps = test_snaps + 1
                e_snapshot_id[test_idx] = test_snaps

        else:
            def _assign_structural(mask: torch.Tensor, start_sid: int):
                idx = mask.nonzero(as_tuple=True)[0]
                m = idx.numel()
                if m == 0:
                    return idx, torch.empty(0, dtype=torch.long, device=device), start_sid
                row = torch.arange(m, device=device)
                snaps = (row // snapshot_size).long() + start_sid
                next_start = int(snaps.max().item()) + 1
                return idx, snaps, next_start

            start = 0
            train_idx, train_snaps, start = _assign_structural(train_mask, start)
            if train_idx.numel() > 0:
                e_snapshot_id[train_idx] = train_snaps

            if val_mask is not None:
                val_idx, val_snaps, start = _assign_structural(val_mask, start)
                if val_idx.numel() > 0:
                    e_snapshot_id[val_idx] = val_snaps
            else:
                val_idx = torch.empty(0, dtype=torch.long, device=device)

            test_idx, test_snaps, _ = _assign_structural(test_mask, start)
            if test_idx.numel() > 0:
                e_snapshot_id[test_idx] = test_snaps

        self._edges["e_snapshot_id"] = e_snapshot_id


    def snapshots(self, split="all", accumulate=True, all_nodes: bool = True):
        """
        Generates subgraphs corresponding to individual snapshots based on snapshot IDs.
        
        Parameters:
        split : str, optional (default="all")
            The data split to use for generating snapshots. Options are:
            - "train": only training edges
            - "val": only validation edges
            - "test": only test edges
            - "all": all edges

        accumulate : bool, optional (default=True)
            If True, each snapshot will contain all edges up to and including the current snapshot ID.
            If False, each snapshot will contain only the edges assigned exactly to that snapshot ID.

        all_nodes : bool, optional (default=True)
            If True, includes all nodes in each snapshot.
            If False, includes only the nodes present in the selected edges for the snapshot.

        Yields:
        Graph
            A subgraph representing a snapshot. Each yielded Graph object includes:
            - A subset of edges (based on snapshot ID and split)
            - A subset of nodes (based on `all_nodes` and node timestamps, if available)

        Notes:
        - Requires that 'e_snapshot_id' is already computed, typically by calling `generate_snapshots`.
        - If nodes have a 'n_snapshot_id' field, they are filtered based on snapshot timing.
        """

        if split == "all":
            split_mask = torch.ones(self.num_edges, dtype=torch.bool)
        elif split == "train":
            split_mask = self.e_train_mask
        elif split == "val":
            split_mask = self.e_val_mask
        elif split == "test":
            split_mask = self.e_test_mask
        else:
            raise ValueError("split must be one of {'train','val','test','all'}")
        
        # only keep edges from split
        e_sids = self._edges["e_snapshot_id"][split_mask]
        snap_ids = torch.unique(e_sids)

        for sid in snap_ids.tolist():

            if accumulate:
                time_mask = self._edges["e_snapshot_id"] <= sid
            else:
                time_mask = self._edges["e_snapshot_id"] == sid
            mask = split_mask & time_mask
            e_idx = mask.nonzero(as_tuple=True)[0]

            if e_idx.numel() == 0:
                continue
            
            edges_sub = {}
            for k, v in self._edges.items():
                if k == "e_pairs":
                    edges_sub[k] = v[:, e_idx]
                elif v.shape[0] == mask.shape[0]:
                    edges_sub[k] = v[e_idx]

            if "n_snapshot_id" in self._nodes:
                if accumulate:
                    n_time_mask = self._nodes["n_snapshot_id"] <= sid
                else:
                    n_time_mask = self._nodes["n_snapshot_id"] == sid
                # if nodes have timestamps we filter these too
                if all_nodes:
                    n_mask = n_time_mask
                else:
                    # nodes present in current edges
                    edge_node_ids = torch.unique(edges_sub["e_pairs"])
                    present = torch.isin(self._nodes["n_id"], edge_node_ids)
                    n_mask = n_time_mask & present
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
                    n_mask = torch.tensor([x.item() in id_map for x in self._nodes["n_id"]])
                    n_idx = n_mask.nonzero(as_tuple=True)[0]
                    for k, v in self._nodes.items():
                        if v.shape[0] == mask.shape[0]:
                            nodes_sub[k] = v[n_idx]

            yield Graph(nodes=nodes_sub, edges=edges_sub)

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
