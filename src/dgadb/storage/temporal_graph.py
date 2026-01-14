import os
import glob
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
        return self.edges.T

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

    def flip_edge_labels(self):
        device = self.edge_labels.device
        self.edge_labels = ((self.edge_labels - torch.tensor(1, device=device))
                            * torch.tensor(-1, device=device))

    def to(self, device: Any, **kwargs):
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
        print("--- TemporalGraphData Summary ---")
        print(f"Device: {self.device}")
        print(f"Number of Nodes: {self.num_nodes}")
        print(f"Number of Edges: {len(self.src)}")
        print(f"  - Train Edges: {self.train_mask.sum().item()}")
        print(f"  - Validation Edges: {self.val_mask.sum().item()}")
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
    def __init__(self, temporal_graph: TemporalGraph, slice_obj: slice):
        self._temporal_graph = temporal_graph
        self._slice = slice_obj
        self._num_edges = len(range(slice_obj.start, slice_obj.stop)[
                              0:slice_obj.step]) if slice_obj else 0

    def __getattr__(self, name: str):
        attr = getattr(self._temporal_graph, name)
        if torch.is_tensor(attr) and attr.size(0) == self._temporal_graph.num_edges:
            return attr[self._slice]

        return attr

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def edge_index(self) -> torch.Tensor:
        return torch.stack([self.src, self.tgt], dim=1).T


class TemporalGraphLoader:
    def __init__(
        self,
        base_directory: str = "processed/",
        metadata_suffix: str = "_meta"
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_directory = base_directory
        self.metadata_suffix = metadata_suffix
        os.makedirs(base_directory, exist_ok=True)
        self.logger.info(
            f"TemporalGraphLoader initialized. Using base directory: {self.base_directory}")

    def save(self, temporal_graph: TemporalGraph) -> str:
        dataset_name = temporal_graph.metadata.get(
            "dataset_name", "unknown-dataset")
        dataset_dir = os.path.join(self.base_directory, dataset_name)

        os.makedirs(dataset_dir, exist_ok=True)

        from .utils import generate_temporal_graph_filename

        prefix = generate_temporal_graph_filename(temporal_graph)
        full_path_prefix = os.path.join(dataset_dir, prefix)

        torch_fn = f"{full_path_prefix}.pt"
        metadata_fn = f"{full_path_prefix}{self.metadata_suffix}.json"

        torch.save(temporal_graph, torch_fn)
        self.logger.info(f"Saved torch object: {torch_fn}")

        with open(metadata_fn, "w") as f:
            json.dump(temporal_graph.metadata, f, indent=4)

        self.logger.info(f"Saved metadata JSON: {metadata_fn}")

        self.logger.info(f"Graph saved successfully.")

        return full_path_prefix

    def load(
        self,
        dataset_name: str,
        anom_type: Optional[str] = None,
        anom_train_ratio: Optional[float] = None,
        anom_val_ratio: Optional[float] = None,
        anom_test_ratio: Optional[float] = None,
        create_if_not_found: bool = False,
        device: Device = None,
        **kwargs
    ) -> TemporalGraph:
        search_criteria = {
            "dataset_name": dataset_name,
            "anomaly_type": anom_type,
            "anomaly_ratios": (
                anom_train_ratio or 0.0,
                anom_val_ratio or 0.0,
                anom_test_ratio or 0.0
            ),
            "anomaly_generation_parameters": kwargs
        }
        self.logger.info(
            f"Searching for graph with criteria: {search_criteria}")

        if anom_type is None:
            if any(r for r in [anom_train_ratio, anom_val_ratio, anom_test_ratio]):
                raise ValueError(
                    "Cannot specify non-zero anomaly ratios when 'anom_type' is None.")
            if kwargs:
                raise ValueError(
                    "Cannot specify generation parameters (kwargs) when 'anom_type' is None.")
        else:
            from src.dgadb.preprocessing import get_canonical_anomaly_type

            anom_type = get_canonical_anomaly_type(anom_type)

        search_pattern = os.path.join(
            self.base_directory, dataset_name, f"{dataset_name}*{self.metadata_suffix}.json")
        possible_files = glob.glob(search_pattern)

        matches = []
        for meta_path in possible_files:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("dataset_name") != dataset_name:
                continue

            is_injected = meta.get("anomaly_injection", {}) \
                .get("is_injected", False)

            if anom_type is None:
                # Looking for clean graph
                if is_injected:
                    continue  # Skip anomalous

                matches.append(meta_path)

            else:
                if not is_injected:
                    continue  # Skip clean

                if meta["anomaly_injection"].get("type") != anom_type:
                    continue

                splits_meta = meta["anomaly_injection"].get("splits", {})
                if splits_meta.get("train", {}).get("ratio") != anom_train_ratio:
                    continue
                if splits_meta.get("val", {}).get("ratio") != anom_val_ratio:
                    continue
                if splits_meta.get("test", {}).get("ratio") != anom_test_ratio:
                    continue

                gen_params = meta.get("anomaly_injection", {}) \
                    .get("generation_parameters", {})
                if any(gen_params.get(key) != value for key, value in kwargs.items()):
                    continue

                matches.append(meta_path)

        if len(matches) == 0:
            if create_if_not_found:
                self.logger.info(
                    "Could not find a matching graph for the specified criteria. Loading from raw data.")

                from src.dgadb.preprocessing import Pipeline

                pipeline = Pipeline.from_config(dataset_name)
                temmporal_graph = pipeline.run().to_temporal_graph()
                if device:
                    temmporal_graph = temmporal_graph.to(device)

                if anom_type is None:
                    return temmporal_graph

                from src.dgadb.preprocessing import AnomalyInjector

                anom_injector = AnomalyInjector(temmporal_graph)
                anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
                    anom_type, anom_train_ratio or 0, anom_val_ratio or 0, anom_test_ratio or 0, **kwargs)

                return anomalous_temporal_graph

            else:
                raise FileNotFoundError(
                    f"Could not find a matching graph for the specified criteria.")

        if len(matches) > 1:
            raise ValueError(
                f"Found multiple matching graphs. Refine your search criteria.\n"
                f"Matches found: {matches}"
            )

        matched_meta_path = matches[0]
        path_prefix = matched_meta_path.replace(
            f"{self.metadata_suffix}.json", "")

        self.logger.info(
            f"Found matching graph: {os.path.basename(path_prefix)}.pt")

        return torch.load(f"{path_prefix}.pt")
