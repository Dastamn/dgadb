import logging
import torch
from dataclasses import dataclass
from typing import Iterator, Literal, Optional
from .temporal_graph import TemporalGraph, TemporalGraphView

_SNAPSHOTTING_STRATEGIES: list[str] = ["window", "event"]


@dataclass
class TemporalGraphSnapshot:
    snapshot_id: int
    current: TemporalGraphView
    cumulative: TemporalGraphView

    # @property
    # def num_cumulative_nodes(self):
    #     return self.cumulative.num_nodes

    # def __repr__(self) -> str:
    #     return (
    #         f"{self.__class__.__name__}(snapshot_id={self.snapshot_id}, "
    #         f"current_edges={self.current.num_edges}, "
    #         f"cumulative_nodes={self.num_cumulative_nodes}, "
    #         f"cumulative_edges={self.cumulative.num_edges}"
    #         ")"
    #     )


class TemporalGraphSnapshotLoader:
    def __init__(
        self,
        data: TemporalGraph,
        strategy: Literal["window", "event"] = "window",
        split: Optional[Literal["train", "val", "test"]] = None,
        **kwargs
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = data
        self.strategy = strategy
        self.split = split
        self.kwargs = kwargs
        self._split_start_i: int = 0
        self._num_split_edges: int = self.data.num_edges

        if self.split is not None:
            if self.split not in ['train', 'val', 'test']:
                raise ValueError(
                    "Split must be one of 'train', 'val' or 'test'.")

            split_mask_name = f"{self.split}_mask"
            split_mask: torch.Tensor = getattr(self.data, split_mask_name)
            if split_mask is None:
                raise RuntimeError(
                    f"Split mask '{split_mask_name}' not found in graph object.")

            if not split_mask.any():
                self.logger.warning(f"'{self.split}' split is empty.")
                self._num_split_edges = 0
            else:
                split_indices = torch.where(split_mask)[0]
                self._split_start_i = int(split_indices[0].item())
                self._num_split_edges = len(split_indices)

        self._snapshot_global_indices = self._compute_snapshot_indices()
        self.reset()

    def _slice_data(self, indices: torch.Tensor | slice) -> TemporalGraph:
        sliced_attrs = {
            key: value[indices]
            for key, value in self.data.__dict__.items()
            if torch.is_tensor(value) and value.size(0) == self.data.num_edges
        }

        return TemporalGraph(
            **sliced_attrs,
            node_attr=self.data.node_attr,
            node_labels=self.data.node_labels
        )

    def reset(self):
        self._current_snapshot_num = 0

    def _compute_snapshot_indices(self) -> list[tuple[int, int]]:
        indices_list = []

        if self._num_split_edges == 0:
            return indices_list

        if self.strategy == 'window':
            window_size = self.kwargs.get('window_size')
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError(
                    "Strategy 'window' requires a positive integer 'window_size'.")

            pos_in_split = 0
            while pos_in_split < self._num_split_edges:
                start_in_split = pos_in_split
                end_in_split = min(pos_in_split + window_size,
                                   self._num_split_edges)

                global_start = self._split_start_i + start_in_split
                global_end = self._split_start_i + end_in_split
                indices_list.append((global_start, global_end))

                pos_in_split = end_in_split

            return indices_list

        if self.strategy == "event":
            global_split_end = self._split_start_i + self._num_split_edges
            src_in_split = self.data.src[self._split_start_i:global_split_end]

            if src_in_split.numel() < 2:
                return [(self._split_start_i, global_split_end)]

            change_mask = src_in_split[1:] != src_in_split[:-1]
            change_indices_in_split = torch.where(change_mask)[0] + 1

            device = self.data.device
            split_points_in_split = torch.cat([
                torch.tensor([0], device=device),
                change_indices_in_split,
                torch.tensor([self._num_split_edges], device=device)
            ])

            global_split_points = self._split_start_i + split_points_in_split
            starts = global_split_points[:-1]
            ends = global_split_points[1:]

            return list(zip(starts.tolist(), ends.tolist()))

        raise ValueError(
            f"Unknown strategy: '{self.strategy}'. Choose from {_SNAPSHOTTING_STRATEGIES}.")

    def __len__(self) -> int:
        return len(self._snapshot_global_indices)

    def __iter__(self) -> Iterator[TemporalGraphSnapshot]:
        self.reset()
        return self

    def __next__(self) -> TemporalGraphSnapshot:
        if self._current_snapshot_num >= len(self):
            raise StopIteration

        start_i, end_i = self._snapshot_global_indices[self._current_snapshot_num]
        current_slice = slice(start_i, end_i)
        cumulative_slice = slice(0, end_i)

        current_view = TemporalGraphView(self.data, current_slice)
        cumulative_view = TemporalGraphView(self.data, cumulative_slice)

        snapshot = TemporalGraphSnapshot(
            snapshot_id=self._current_snapshot_num,
            current=current_view,
            cumulative=cumulative_view
        )

        self._current_snapshot_num += 1
        return snapshot
