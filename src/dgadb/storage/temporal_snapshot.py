import torch
from dataclasses import dataclass
from typing import Iterator, Literal, Optional
from .temporal_graph_data import TemporalGraphData

_SNAPSHOTTING_STRATEGIES: list[str] = ["window", "event"]


@dataclass
class TemporalSnapshot:
    snapshot_id: int
    current: TemporalGraphData
    cumulative: TemporalGraphData

    @property
    def num_cumulative_nodes(self):
        return self.cumulative.num_nodes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(snapshot_idx={self.snapshot_id}, "
            f"current_edges={self.current.num_edges}, "
            f"cumulative_nodes={self.num_cumulative_nodes}, "
            f"cumulative_edges={self.cumulative.num_edges}"
            ")"
        )


class TemporalGraphSnapshotLoader:
    def __init__(self, data: TemporalGraphData, strategy: Literal["window", "event"] = "window", window_size: Optional[int] = 1000):
        self.data = data
        self.strategy = strategy
        self.window_size = window_size

        self._snapshot_indices: list[tuple[int, int]] = \
            self._compute_snapshot_indices()

        self.reset()

    def reset(self):
        self._current_snapshot_num = 0

    def _compute_snapshot_indices(self) -> list[tuple[int, int]]:
        indices = []
        pos = 0
        num_edges = self.data.num_edges

        if self.strategy == 'window':
            if self.window_size is None or self.window_size <= 0:
                raise ValueError(
                    "Strategy 'window' requires a positive integer 'window_size'.")
            while pos < num_edges:
                start = pos
                end = min(pos + self.window_size, num_edges)
                indices.append((start, end))
                pos = end

        elif self.strategy == 'event':
            if num_edges == 0:
                return []

            if num_edges == 1:
                return [(0, 1)]

            device = self.data.device
            src_nodes = self.data.src
            change_mask = src_nodes[1:] != src_nodes[:-1]
            change_indices = torch.where(change_mask)[0] + 1
            start_indices = torch.cat(
                [torch.tensor([0], device=device), change_indices])
            end_indices = torch.cat(
                [change_indices, torch.tensor([num_edges], device=device)])
            indices = list(zip(start_indices.tolist(), end_indices.tolist()))

        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose from {_SNAPSHOTTING_STRATEGIES}.")

        return indices

    def __len__(self) -> int:
        return len(self._snapshot_indices)

    def __iter__(self) -> Iterator[TemporalSnapshot]:
        self.reset()
        return self

    def __next__(self) -> TemporalSnapshot:
        if self._current_snapshot_num >= len(self):
            raise StopIteration

        start_idx, end_idx = self._snapshot_indices[self._current_snapshot_num]

        current_slice = {
            key: value[start_idx:end_idx]
            for key, value in self.data.__dict__.items()
            if torch.is_tensor(value) and value.size(0) == self.data.num_edges
        }
        current_graph = TemporalGraphData(
            **current_slice,
            node_attr=self.data.node_attr,
            node_labels=self.data.node_labels
        )

        cumulative_slice = {
            key: value[0:end_idx]
            for key, value in self.data.__dict__.items()
            if torch.is_tensor(value) and value.size(0) == self.data.num_edges
        }
        cumulative_graph = TemporalGraphData(
            **cumulative_slice,
            node_attr=self.data.node_attr,
            node_labels=self.data.node_labels
        )

        snapshot = TemporalSnapshot(
            snapshot_id=self._current_snapshot_num,
            current=current_graph,
            cumulative=cumulative_graph
        )

        self._current_snapshot_num += 1

        return snapshot
