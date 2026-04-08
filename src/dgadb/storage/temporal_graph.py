import numpy as np
from typing import Literal
import os
import json
import copy
import logging

from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch

from torch.types import Device


@dataclass
class TemporalGraph:
    """Core temporal-graph container used throughout the benchmark.

    Holds all edge-level tensors (source, target, timestamp, message features,
    labels, split masks) plus optional node-level attributes and a free-form
    ``metadata`` dict. Unknown attributes fall through to ``metadata`` via
    ``__getattr__``/``__setattr__`` so downstream code can attach fields like
    ``dataset_name`` or ``variant_name`` without subclassing.

    Attributes:
        src: Source node indices, shape ``[num_edges]``.
        tgt: Target node indices, shape ``[num_edges]``.
        t: Edge timestamps, shape ``[num_edges]``.
        msg: Edge features, shape ``[num_edges, num_edge_features]``.
        edge_labels: Optional binary anomaly labels per edge.
        train_mask: Boolean mask selecting training edges.
        test_mask: Boolean mask selecting test edges.
        val_mask: Optional boolean mask selecting validation edges.
        w: Optional per-edge weight.
        node_attr: Optional node feature matrix.
        node_labels: Optional per-node labels.
        metadata: Free-form dict for dataset name, anomaly-injection info, etc.
    """

    src: torch.Tensor                   # Shape: [num_edges]
    tgt: torch.Tensor                   # Shape: [num_edges]
    t: torch.Tensor                     # Shape: [num_edges], timestamps
    msg: torch.Tensor                   # Shape: [num_edges, num_edge_features]
    edge_labels: Optional[torch.Tensor]  # Shape: [num_edges]

    train_mask: torch.Tensor            # Shape: [num_edges], boolean
    test_mask: torch.Tensor             # Shape: [num_edges], boolean
    val_mask: Optional[torch.Tensor]    # Shape: [num_edges], boolean

    # Shape: [num_edges]
    w: Optional[torch.Tensor] = None
    # Shape: [num_nodes, num_node_features]
    node_attr: Optional[torch.Tensor] = None
    # Shape: [num_nodes]
    node_labels: Optional[torch.Tensor] = None

    metadata: dict = field(default_factory=dict)

    # def __post_init__(self) -> None:
    #     self.check_device()
    #     if getattr(self, "total_num_nodes"):
    #         self.metadata["total_num_nodes"] = self.num_nodes

    @property
    def device(self) -> torch.device:
        return self.check_device()

    @property
    def num_nodes(self) -> int:
        return max(int(self.src.max()), int(self.tgt.max())) + 1

    @property
    def num_edges(self) -> int:
        return self.src.size(0)

    @property
    def edges(self):
        return torch.stack([self.src, self.tgt], dim=1)

    @property
    def edge_index(self) -> torch.Tensor:
        return torch.stack([self.src, self.tgt], dim=0)

    @property
    def adj_matrix_coo(self) -> torch.Tensor:
        device = self.device
        size = self.num_nodes
        if self.num_edges == 0:
            return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.int64),
                                           torch.empty((0,), dtype=torch.float32), size=(size, size), device=device)

        indices = torch.stack([self.src, self.tgt], dim=0)
        values = (self.w if self.w is not None
                  else torch.ones(self.num_edges, dtype=torch.float32, device=device))

        return torch.sparse_coo_tensor(indices, values, size=(size, size), device=device)

    @property
    def adj_matrix_csr(self) -> torch.Tensor:
        return self.adj_matrix_coo.to_sparse_csr()

    @property
    def adj_matrix_dense(self) -> torch.Tensor:
        return self.adj_matrix_coo.to_dense()

    # def flip_edge_labels(self):
    #     device = self.edge_labels.device
    #     self.edge_labels = ((self.edge_labels - torch.tensor(1, device=device))
    #                         * torch.tensor(-1, device=device))

    def to(self, device: Any, **kwargs):
        """Return a copy of this graph with all tensors moved to ``device``.

        Non-tensor attributes (including ``metadata``) are deep-copied. Returns
        ``self`` unchanged if the graph is already on the requested device.
        """
        if self.device == torch.device(device):
            return self

        new_attrs = {}
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                new_attrs[key] = value.to(device, **kwargs)
            else:
                new_attrs[key] = copy.deepcopy(value)  # copy non-tensors

        return self.__class__(**new_attrs)

    def check_device(self) -> torch.device:
        """Return the device shared by all tensor attributes.

        Raises:
            ValueError: If no tensor attributes exist.
            RuntimeError: If tensors are on multiple devices.
        """
        devices = defaultdict(list)
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                devices[value.device].append(key)

        if not devices:
            raise ValueError("No tensor attributes found.")

        if len(devices) > 1:
            raise RuntimeError(
                f"Tensors are on different devices: {devices}")

        return next(iter(devices.keys()))

    def describe(self) -> None:
        """Print a human-readable summary of the graph to stdout."""
        print("--- TemporalGraphData Summary ---")
        print(f"Device: {self.device}")
        print(f"Number of Nodes: {self.num_nodes}")
        print(f"Number of Edges: {len(self.src)}")
        print(f"  - Train Edges: {self.train_mask.sum().item()}")
        print(
            f"  - Validation Edges: {self.val_mask.sum().item() if self.val_mask is not None else None}")
        print(f"  - Test Edges: {self.test_mask.sum().item()}")
        print("\nTensor Attributes:")
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                print(
                    f"  - {key}: shape={list(value.shape)}, dtype={value.dtype}")
        print("\nMetadata:")
        for key, value in self.metadata.items():
            if isinstance(value, (list, set, dict)):
                print(
                    f"  - {key}: <{type(value).__name__} of length {len(value)}>")
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
        return (
            f"{self.__class__.__name__}("
            f"src={self.src.shape}, "
            f"tgt={self.tgt.shape}, "
            f"t={self.t.shape}, "
            f"msg={self.t.shape}, "
            f"node_attr={self.node_attr.shape if self.node_attr is not None else None}, "
            f"device={self.device}"
            ")"
        )


class TemporalGraphView:
    """Zero-copy view over a subset of edges of a :class:`TemporalGraph`.

    Attribute access is forwarded to the underlying graph; tensors whose first
    dimension equals the number of edges are transparently indexed by the
    stored ``indices``. Used by the snapshot loader to avoid copying data.
    """

    def __init__(self, temporal_graph: TemporalGraph, indices: slice | torch.Tensor):
        self._temporal_graph = temporal_graph
        self._indices = indices
        
        if isinstance(indices, slice):
            self._num_edges = len(range(*indices.indices(temporal_graph.num_edges)))
        else:
            self._num_edges = indices.size(0)

    def __getattr__(self, name: str):
        attr = getattr(self._temporal_graph, name)
        if torch.is_tensor(attr) and attr.dim() > 0 and attr.size(0) == self._temporal_graph.num_edges:
            return attr[self._indices]

        return attr

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def edge_index(self) -> torch.Tensor:
        return torch.stack([self.src, self.tgt], dim=0)


class TemporalGraphLoaderNew:
    """Variant-directory loader for processed :class:`TemporalGraph` objects.

    Stores each (dataset, anomaly variant) under a dedicated directory
    (``base_directory/<dataset>/<variant_name>/``) containing ``data.pt`` and
    ``metadata.json``. Used by :mod:`dgadb.experiment.runner`; coexists with
    the legacy :class:`TemporalGraphLoader` until the follow-up migration.

    Args:
        base_directory: Root directory for processed variants.
    """

    def __init__(self, base_directory: str = "processed") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_dir = base_directory
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_variant_name(self, anom_type: Optional[str], tr: float, v: float, te: float, dur: float) -> Optional[str]:
        if anom_type is None:
            return None

        assert dur is not None
        return f"{anom_type}_tr{tr}_v{v}_te{te}_dur{dur}"

    def _prepare_json_meta(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            new_dict = {k: self._prepare_json_meta(v) for k, v in obj.items()}
            if "anomaly_group_ids" in obj and torch.is_tensor(obj["anomaly_group_ids"]):
                ids = obj["anomaly_group_ids"]
                new_dict["anomaly_summary"] = {
                    "total_anomaly_groups": int(ids.max().item()),
                    "total_anomalous_edges": int((ids > 0).sum().item())
                }
            return new_dict

        elif isinstance(obj, list):
            return [self._prepare_json_meta(v) for v in obj]
        elif torch.is_tensor(obj):
            return f"<Tensor: shape={list(obj.shape)}, dtype={obj.dtype}>"
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        else:
            return obj

    def save(self, tg: TemporalGraph, variant_dir: str):
        """Persist ``tg`` to ``variant_dir`` as ``data.pt`` + ``metadata.json``."""
        os.makedirs(variant_dir, exist_ok=True)
        torch.save(tg, os.path.join(variant_dir, "data.pt"))
        json_meta = self._prepare_json_meta(tg.metadata)
        with open(os.path.join(variant_dir, "metadata.json"), "w") as f:
            json.dump(json_meta, f, indent=4)

        self.logger.info(f"Saved data.pt and meta.json to {variant_dir}")

    def load(
        self,
        dataset_name: str,
        anom_type: Optional[Literal["random", "burst",
                                    "clique", "path", "bridge"]] = None,
        anom_train_ratio: float = 0.0,
        anom_val_ratio: float = 0.0,
        anom_test_ratio: float = 0.0,
        duration: float | Literal["small", "medium", "large"] = "medium",
        create_if_not_found: bool = False
    ) -> TemporalGraph:
        """Load (or generate) a variant of ``dataset_name``.

        Looks up ``base_directory/<dataset>/<variant>/data.pt`` where
        ``variant`` is derived from the anomaly configuration. When no match
        exists and ``create_if_not_found`` is ``True``, runs the preprocessing
        pipeline for the clean graph and the anomaly injector for the variant.

        Args:
            dataset_name: Name of the dataset.
            anom_type: Anomaly type; ``None`` returns the clean graph.
            anom_train_ratio: Anomaly ratio in the train split.
            anom_val_ratio: Anomaly ratio in the validation split.
            anom_test_ratio: Anomaly ratio in the test split.
            duration: Anomaly duration, either a literal bucket
                (``"small"``/``"medium"``/``"large"``) or a float rate.
            create_if_not_found: Run the preprocessing pipeline and anomaly
                injector when no cached variant is found.

        Returns:
            The matched or newly created :class:`TemporalGraph`.

        Raises:
            FileNotFoundError: When no cached variant exists and
                ``create_if_not_found`` is ``False``.
        """
        if isinstance(duration, float):
            duration_rate = duration
        else:
            from dgadb.preprocessing.anomaly_injection import _ANOMALY_DURATION_TYPE_MAP
            
            duration_rate = _ANOMALY_DURATION_TYPE_MAP[str(duration)]

        variant_name = self._get_variant_name(
            anom_type, anom_train_ratio, anom_val_ratio, anom_test_ratio, duration_rate)
        variant_dir = os.path.join(
            self.base_dir, dataset_name, variant_name or "clean")
        data_path = os.path.join(variant_dir, "data.pt")

        if os.path.exists(data_path):
            self.logger.info(f"Loading existing graph from {variant_dir}")
            return torch.load(data_path, weights_only=False)

        if not create_if_not_found:
            raise FileNotFoundError(f"No graph found at {variant_dir}")

        self.logger.info(
            f"Requested variant not found. Creating {dataset_name} ({anom_type})...")

        clean_dir = os.path.join(self.base_dir, dataset_name, "clean")
        clean_path = os.path.join(clean_dir, "data.pt")

        if os.path.exists(clean_path):
            tg = torch.load(clean_path, weights_only=False)
        else:
            from dgadb.preprocessing import Pipeline
            self.logger.info(
                f"Clean graph not found. Running pipeline for {dataset_name}...")
            pipeline = Pipeline.from_config(dataset_name)
            tg = pipeline.run().to_temporal_graph()
            tg.metadata["dataset_name"] = dataset_name
            self.save(tg, clean_dir)

        if anom_type is not None:
            from dgadb.preprocessing.anomaly_injection import AnomalyInjector
            injector = AnomalyInjector(tg, cache_dir=self.base_dir)
            tg = injector.generate_anomalous_samples(
                anom_type=anom_type,
                train_ratio=anom_train_ratio,
                val_ratio=anom_val_ratio,
                test_ratio=anom_test_ratio,
                duration=duration_rate
            )
            tg.metadata["variant_name"] = variant_name
            self.save(tg, variant_dir)

        return tg
