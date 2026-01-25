import numpy as np
from typing import Literal
from pathlib import Path
import json
import copy
import logging

from typing import Any, Optional
from dataclasses import dataclass, field, fields
from collections import defaultdict

import torch
from safetensors.torch import save_file, load_file

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

    # def flip_edge_labels(self):
    #     device = self.edge_labels.device
    #     self.edge_labels = ((self.edge_labels - torch.tensor(1, device=device))
    #                         * torch.tensor(-1, device=device))

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
        self.base_directory = Path(base_directory)
        self.metadata_suffix = metadata_suffix
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"TemporalGraphLoader initialized. Using base directory: {self.base_directory}")

    def save(self, temporal_graph: TemporalGraph) -> Path:
        dataset_name = temporal_graph.metadata.get(
            "dataset_name", "unknown-dataset")
        dataset_dir = self.base_directory / dataset_name

        dataset_dir.mkdir(parents=True, exist_ok=True)

        from .utils import generate_temporal_graph_filename

        prefix = generate_temporal_graph_filename(temporal_graph)
        full_path_prefix = dataset_dir / prefix

        torch_path = full_path_prefix.with_suffix(".pt")
        metadata_path = full_path_prefix.parent / f"{full_path_prefix.name}{self.metadata_suffix}.json"

        torch.save(temporal_graph, torch_path)
        self.logger.info(f"Saved torch object: {torch_path}")

        metadata_path.write_text(json.dumps(temporal_graph.metadata, indent=4))
        self.logger.info(f"Saved metadata JSON: {metadata_path}")

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
            from dgadb.preprocessing import get_canonical_anomaly_type

            anom_type = get_canonical_anomaly_type(anom_type)

        search_dir = self.base_directory / dataset_name
        possible_files = list(search_dir.glob(f"{dataset_name}*{self.metadata_suffix}.json"))

        matches = []
        for meta_path in possible_files:
            meta = json.loads(meta_path.read_text())

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

                from dgadb.preprocessing import Pipeline

                pipeline = Pipeline.from_config(dataset_name)
                temmporal_graph = pipeline.run().to_temporal_graph()
                if device:
                    temmporal_graph = temmporal_graph.to(device)

                if anom_type is None:
                    return temmporal_graph

                from dgadb.preprocessing import AnomalyInjector

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
        # Remove the metadata suffix to get the base path
        base_name = matched_meta_path.name.replace(f"{self.metadata_suffix}.json", "")
        torch_path = matched_meta_path.parent / f"{base_name}.pt"

        self.logger.info(f"Found matching graph: {torch_path.name}")

        return torch.load(torch_path, weights_only=False)


class TemporalGraphLoaderNew:
    def __init__(self, base_directory: str = "processed") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_dir = Path(base_directory)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_variant_name(self, anom_type: Optional[str], tr: float, v: float, te: float, dur: Optional[str]) -> Optional[str]:
        if anom_type is None:
            return None

        assert dur is not None
        return f"{anom_type}_tr{tr}_v{v}_te{te}_{dur}"

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

    def _extract_tensors(self, tg: TemporalGraph) -> dict[str, torch.Tensor]:
        """Extract all tensor attributes from TemporalGraph for safetensors serialization."""
        tensors = {}

        # Extract dataclass field tensors
        for f in fields(tg):
            value = getattr(tg, f.name)
            if torch.is_tensor(value):
                # safetensors requires contiguous tensors
                tensors[f.name] = value.contiguous()

        # Extract tensor values from metadata (e.g., anomaly_group_ids)
        self._extract_metadata_tensors(tg.metadata, "metadata", tensors)

        return tensors

    def _extract_metadata_tensors(
        self, obj: Any, prefix: str, tensors: dict[str, torch.Tensor]
    ) -> None:
        """Recursively extract tensors from metadata dict."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._extract_metadata_tensors(value, f"{prefix}.{key}", tensors)
        elif torch.is_tensor(obj):
            tensors[prefix] = obj.contiguous()

    def _prepare_metadata_for_json(self, metadata: dict) -> dict:
        """Prepare metadata for JSON serialization by removing tensors."""
        result = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                result[key] = self._prepare_metadata_for_json(value)
            elif torch.is_tensor(value):
                # Store a marker that this was a tensor (for reconstruction)
                result[key] = {"__tensor__": True, "shape": list(value.shape), "dtype": str(value.dtype)}
            elif isinstance(value, (np.integer, int)):
                result[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                result[key] = float(value)
            else:
                result[key] = value
        return result

    def save(self, tg: TemporalGraph, variant_dir: Path):
        variant_dir = Path(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Save tensors using safetensors
        tensors = self._extract_tensors(tg)
        save_file(tensors, variant_dir / "data.safetensors")

        # Save metadata as JSON (with tensor markers for reconstruction)
        json_meta = self._prepare_metadata_for_json(tg.metadata)
        (variant_dir / "metadata.json").write_text(json.dumps(json_meta, indent=4))

        self.logger.info(f"Saved data.safetensors and metadata.json to {variant_dir}")

    def _reconstruct_metadata(self, json_meta: dict, tensors: dict[str, torch.Tensor], prefix: str = "metadata") -> dict:
        """Reconstruct metadata dict, replacing tensor markers with actual tensors."""
        result = {}
        for key, value in json_meta.items():
            tensor_key = f"{prefix}.{key}"
            if isinstance(value, dict):
                if value.get("__tensor__") is True:
                    # This was a tensor, retrieve from tensors dict
                    result[key] = tensors[tensor_key]
                else:
                    result[key] = self._reconstruct_metadata(value, tensors, tensor_key)
            else:
                result[key] = value
        return result

    def _load_from_safetensors(self, variant_dir: Path) -> TemporalGraph:
        """Load a TemporalGraph from safetensors format."""
        variant_dir = Path(variant_dir)
        tensors = load_file(variant_dir / "data.safetensors")
        json_meta = json.loads((variant_dir / "metadata.json").read_text())

        # Reconstruct metadata with tensors
        metadata = self._reconstruct_metadata(json_meta, tensors)

        # Build TemporalGraph from tensors
        return TemporalGraph(
            src=tensors["src"],
            tgt=tensors["tgt"],
            t=tensors["t"],
            msg=tensors["msg"],
            edge_labels=tensors.get("edge_labels"),
            train_mask=tensors["train_mask"],
            test_mask=tensors["test_mask"],
            val_mask=tensors.get("val_mask"),
            w=tensors.get("w"),
            node_attr=tensors.get("node_attr"),
            node_labels=tensors.get("node_labels"),
            metadata=metadata,
        )

    def _load_from_legacy(self, data_path: Path) -> TemporalGraph:
        """Load from legacy torch.save format (for backwards compatibility)."""
        return torch.load(data_path, weights_only=False)

    def load(
        self,
        dataset_name: str,
        anom_type: Optional[Literal["random", "burst",
                                    "clique", "path", "bridge"]] = None,
        anom_train_ratio: float = 0.0,
        anom_val_ratio: float = 0.0,
        anom_test_ratio: float = 0.0,
        duration_type: Literal["small", "medium", "large"] = "medium",
        create_if_not_found: bool = False
    ) -> TemporalGraph:
        variant_name = self._get_variant_name(
            anom_type, anom_train_ratio, anom_val_ratio, anom_test_ratio, duration_type)
        variant_dir = self.base_dir / dataset_name / (variant_name or "clean")

        safetensors_path = variant_dir / "data.safetensors"
        legacy_path = variant_dir / "data.pt"

        # Try safetensors first, fall back to legacy format
        if safetensors_path.exists():
            self.logger.info(f"Loading existing graph from {variant_dir} (safetensors)")
            return self._load_from_safetensors(variant_dir)
        elif legacy_path.exists():
            self.logger.info(f"Loading existing graph from {variant_dir} (legacy format)")
            return self._load_from_legacy(legacy_path)

        if not create_if_not_found:
            raise FileNotFoundError(f"No graph found at {variant_dir}")

        self.logger.info(
            f"Requested variant not found. Creating {dataset_name} ({anom_type})...")

        clean_dir = self.base_dir / dataset_name / "clean"
        clean_safetensors_path = clean_dir / "data.safetensors"
        clean_legacy_path = clean_dir / "data.pt"

        if clean_safetensors_path.exists():
            tg = self._load_from_safetensors(clean_dir)
        elif clean_legacy_path.exists():
            tg = self._load_from_legacy(clean_legacy_path)
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
                duration_type=duration_type
            )
            tg.metadata["variant_name"] = variant_name
            self.save(tg, variant_dir)

        return tg
