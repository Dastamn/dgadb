import copy
import torch
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TemporalGraphData:
    src: torch.Tensor                   # Shape: [num_edges]
    tgt: torch.Tensor                   # Shape: [num_edges]
    t: torch.Tensor                     # Shape: [num_edges], timestamps
    msg: torch.Tensor                   # Shape: [num_edges, num_edge_features]
    edge_labels: torch.Tensor           # Shape: [num_edges]

    train_mask: torch.Tensor            # Shape: [num_edges], boolean
    val_mask: torch.Tensor              # Shape: [num_edges], boolean
    test_mask: torch.Tensor             # Shape: [num_edges], boolean

    # Shape: [num_edges], edge weights
    w: Optional[torch.Tensor] = None
    # Shape: [num_nodes, num_node_features]
    node_attr: Optional[torch.Tensor] = None
    # Shape: [num_nodes]
    node_labels: Optional[torch.Tensor] = None

    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.check_device()

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

    def to(self, device: Any, **kwargs):
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
