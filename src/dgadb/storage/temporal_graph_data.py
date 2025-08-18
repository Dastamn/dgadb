import torch
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TemporalGraphData:
    src: torch.Tensor                   # Shape: [num_edges]
    tgt: torch.Tensor                   # Shape: [num_edges]
    t: torch.Tensor                     # Shape: [num_edges], timestamps
    msg: torch.Tensor                   # Shape: [num_edges, num_edge_features]
    edge_labels: torch.Tensor            # Shape: [num_edges]

    node_attr: Optional[torch.Tensor]   # Shape: [num_nodes, num_node_features]
    node_labels: Optional[torch.Tensor]  # Shape: [num_nodes]

    train_mask: torch.Tensor            # Shape: [num_edges], boolean
    val_mask: torch.Tensor              # Shape: [num_edges], boolean
    test_mask: torch.Tensor             # Shape: [num_edges], boolean

    metadata: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return max(int(self.src.max()), int(self.tgt.max())) + 1

    @property
    def num_edges(self) -> int:
        return self.src.size(0)

    def describe(self) -> None:
        print("--- TemporalGraphData Summary ---")
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
