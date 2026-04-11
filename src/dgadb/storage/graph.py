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
    """Legacy node/edge-dict graph container.

    Stores nodes and edges as dictionaries keyed by names from the
    ``N_KEYS`` and ``E_KEYS`` schemas. Supports per-snapshot iteration via
    :meth:`generate_snapshots` and :meth:`snapshots`. Retained for older
    methods that have not migrated to :class:`TemporalGraph`.

    Args:
        nodes: Dict of node tensors keyed by ``N_KEYS`` entries.
        edges: Dict of edge tensors keyed by ``E_KEYS`` entries.
        window_size: Optional snapshot window size hint.
    """

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
            self._nodes["n_id"] = node_ids.sort().values

        if "n_feat" not in self._nodes:
            self._nodes["n_feat"] = torch.eye(node_ids.size(0))

    def _validate_and_set(self, key: str, value: torch.Tensor, is_node: bool):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{'Node' if is_node else 'Edge'} key '{key}' must be a torch.Tensor")
        if is_node:
            if key not in N_KEYS:
                raise ValueError(f"Invalid node key: {key}")
            self._nodes[key] = value
        else:
            if key not in E_KEYS:
                raise ValueError(f"Invalid edge key: {key}")
            self._edges[key] = value

    def __getattr__(self, name: str) -> Any:
        _nodes = object.__getattribute__(self, "_nodes")
        _edges = object.__getattribute__(self, "_edges")

        if name in _nodes:
            return _nodes[name]
        if name in _edges:
            return _edges[name]
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
        """Return a shallow copy of the node tensor dictionary."""
        return self._nodes.copy()

    def edge_dict(self) -> Dict[str, torch.Tensor]:
        """Return a shallow copy of the edge tensor dictionary."""
        return self._edges.copy()

    def to(self, device: torch.device):
        """Move all stored node and edge tensors to ``device`` in place."""
        for k in self._nodes:
            self._nodes[k] = self._nodes[k].to(device)
        for k in self._edges:
            self._edges[k] = self._edges[k].to(device)
        return self

    def generate_snapshots(self, snapshot_size: int, temporal_snapshots: bool = False):
        """Assign snapshot IDs to edges via temporal or structural slicing.

        Modifies the graph in-place by setting ``e_snapshot_id`` on every edge.
        When ``e_val_mask`` is present, validation and test snapshot IDs are
        shifted so they never overlap with preceding splits.

        Args:
            snapshot_size: Window duration (temporal mode) or number of edges
                per chunk (structural mode).
            temporal_snapshots: If ``True``, compute snapshot IDs from
                ``e_timestamp`` divided by ``snapshot_size``. If ``False``,
                group edges into fixed-size chunks within each split.
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
            train_idx, train_snaps, start = _assign_structural(
                train_mask, start)
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
        """Yield per-snapshot subgraphs derived from pre-computed snapshot IDs.

        Requires that :meth:`generate_snapshots` has been called first. If
        nodes have an ``n_snapshot_id`` field, they are filtered consistently
        with the current snapshot ID.

        Args:
            split: Which split's edges to iterate over (``"train"``,
                ``"val"``, ``"test"``, or ``"all"``).
            accumulate: If ``True``, each snapshot includes all edges up to
                and including the current snapshot ID. If ``False``, only
                edges with exactly the current snapshot ID are included.
            all_nodes: If ``True``, include all nodes in each snapshot.
                If ``False``, restrict nodes to those present in the
                selected edges.

        Yields:
            A :class:`Graph` subgraph for each snapshot ID in ``split``.

        Raises:
            ValueError: If ``split`` is not one of the accepted values.
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
            raise ValueError(
                "split must be one of {'train','val','test','all'}")

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
                nodes_sub = {k: v[n_idx] for k, v in self._nodes.items(
                ) if v.shape[0] == n_mask.shape[0]}

            else:
                if all_nodes:
                    # include all nodes
                    nodes_sub = self._nodes
                else:
                    # just include nodes present in current edges
                    edge_node_ids = torch.unique(edges_sub["e_pairs"])
                    id_map = {id.item(): i for i,
                              id in enumerate(edge_node_ids)}
                    nodes_sub = {}
                    n_mask = torch.tensor(
                        [x.item() in id_map for x in self._nodes["n_id"]])
                    n_idx = n_mask.nonzero(as_tuple=True)[0]
                    for k, v in self._nodes.items():
                        if v.shape[0] == mask.shape[0]:
                            nodes_sub[k] = v[n_idx]

            yield Graph(nodes=nodes_sub, edges=edges_sub)

    @property
    def num_nodes(self) -> int:
        """Number of nodes recorded in this graph."""
        if "n_id" in self._nodes:
            return self._nodes["n_id"].shape[0]
        return 0

    @property
    def num_edges(self) -> int:
        """Number of edges recorded in this graph."""
        if "e_id" in self._edges:
            return self._edges["e_id"].shape[0]
        return 0

    @property
    def num_node_types(self) -> int:
        """Number of distinct node types; returns 1 when ``n_type`` is absent."""
        if "n_type" in self._nodes:
            return int(self._nodes["n_type"].unique().numel())
        return 1

    @property
    def num_edge_types(self) -> int:
        """Number of distinct edge types; returns 1 when ``e_type`` is absent."""
        if "e_type" in self._edges:
            return int(self._edges["e_type"].unique().numel())
        return 1

    @property
    def node_types(self):
        """Unique node type values as a CPU tensor."""
        if "n_type" in self._nodes:
            return self._nodes["n_type"].unique().cpu()
        return torch.tensor([0])

    @property
    def num_snapshots(self) -> int:
        """Maximum snapshot ID across nodes and edges; 0 when none are set."""
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
        """Unique edge type values as a CPU tensor."""
        if "e_type" in self._edges:
            return self._edges["e_type"].unique().cpu()
        return torch.tensor([0])

    @property
    def node_feature_dim(self) -> int:
        """Dimensionality of node features; 0 when ``n_feat`` is absent."""
        if "n_feat" in self._nodes:
            return self._nodes["n_feat"].shape[1]
        return 0

    @property
    def edge_feature_dim(self) -> int:
        """Dimensionality of edge features; 0 when ``e_feat`` is absent."""
        if "e_feat" in self._edges:
            return self._edges["e_feat"].shape[1]
        return 0


# TODO make it so that different edge/node types can have differet feature dims
