import logging
import torch
from dataclasses import dataclass
from typing import Iterator, Literal, Optional
from .temporal_graph import TemporalGraph, TemporalGraphView

_SNAPSHOTTING_STRATEGIES: list[str] = ["window", "event"]


@dataclass
class TemporalGraphSnapshot:
    """A single temporal snapshot fed to a model during training or inference.

    Attributes:
        snapshot_id: Sequential index of the snapshot within the loader.
        current: Edges belonging to the current window.
        cumulative: Optional view containing all edges up to and including
            the current window (used by methods that require running history).
    """

    snapshot_id: int
    current: TemporalGraph | TemporalGraphView
    cumulative: Optional[TemporalGraph | TemporalGraphView]

    def __len__(self) -> int:
        return self.current.src.shape[0]


class TemporalGraphSnapshotLoader:
    """Iterates over a :class:`TemporalGraph` as a sequence of snapshots.

    Splits the edge stream into snapshots according to a strategy:

    - ``"window"``: fixed-size batches of ``window_size`` edges.
    - ``"event"``: groups consecutive edges sharing the same source node.

    The loader honours train/val/test masks via ``split`` and can optionally
    yield a cumulative view alongside the current snapshot.

    Args:
        data: The temporal graph to iterate over.
        strategy: Snapshot strategy (``"window"`` or ``"event"``).
        split: Restrict iteration to a single split (``"train"``/``"val"``/``"test"``).
        copy_on_load: If ``True``, copy tensors per snapshot instead of viewing.
        include_cumulative: If ``True``, also expose all preceding edges.
        **kwargs: Strategy parameters (e.g. ``window_size`` for ``"window"``).
    """

    def __init__(
        self,
        data: TemporalGraph,
        strategy: Literal["window", "event"] = "window",
        split: Optional[Literal["train", "val", "test"]] = None,
        copy_on_load: bool = False,
        include_cumulative: bool = False,
        **kwargs
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = data
        self.strategy = strategy
        self.split = split
        self.copy_on_load = copy_on_load
        self.include_cumulative = include_cumulative
        self.kwargs = kwargs
        
        self._snapshot_indices: list[torch.Tensor] = self._compute_snapshot_indices()
        self.reset()

    @property
    def total_num_nodes(self):
        """Total number of nodes in the underlying full graph."""
        return self.data.num_nodes

    def _slice_data(self, indices: torch.Tensor) -> TemporalGraph:
        sliced_attrs = {}
        for key, value in self.data.__dict__.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == self.data.num_edges:
                sliced_attrs[key] = value[indices].clone()
        
        return TemporalGraph(
            **sliced_attrs,
            node_attr=self.data.node_attr,
            node_labels=self.data.node_labels
        )

    def reset(self):
        """Rewind the iterator to the first snapshot."""
        self._current_snapshot_num = 0

    def _compute_snapshot_indices(self) -> list[torch.Tensor]:
        indices_list = []

        if self.split is not None:
            split_mask = getattr(self.data, f"{self.split}_mask")
            if split_mask is None or not split_mask.any():
                return []
            all_split_indices = torch.where(split_mask)[0]
        else:
            all_split_indices = torch.arange(self.data.num_edges, device=self.data.device)

        num_split_edges = len(all_split_indices)

        if self.strategy == 'window':
            window_size = self.kwargs.get('window_size')
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError("Strategy 'window' requires a positive integer 'window_size'.")

            pos = 0
            while pos < num_split_edges:
                batch_indices = all_split_indices[pos : pos + window_size]
                indices_list.append(batch_indices)
                pos += window_size

            return indices_list

        if self.strategy == "event":
            if num_split_edges < 1:
                return []
            
            src_in_split = self.data.src[all_split_indices]
            
            change_mask = torch.cat([
                torch.tensor([True], device=self.data.device),
                src_in_split[1:] != src_in_split[:-1]
            ])
            change_points = torch.where(change_mask)[0]
            
            for i in range(len(change_points)):
                start_idx = change_points[i]
                end_idx = change_points[i+1] if (i+1) < len(change_points) else num_split_edges
                indices_list.append(all_split_indices[start_idx:end_idx])

            return indices_list

        raise ValueError(f"Unknown strategy: '{self.strategy}'. Choose from {_SNAPSHOTTING_STRATEGIES}.")

    def __len__(self) -> int:
        return len(self._snapshot_indices)

    def __iter__(self) -> Iterator[TemporalGraphSnapshot]:
        self.reset()
        return self

    def __next__(self) -> TemporalGraphSnapshot:
        if self._current_snapshot_num >= len(self):
            raise StopIteration

        current_indices = self._snapshot_indices[self._current_snapshot_num]

        if self.copy_on_load:
            current = self._slice_data(current_indices)
            cumulative = None
            if self.include_cumulative:
                # Cumulative is everything from edge 0 up to the maximum index in the current batch
                max_idx = int(current_indices.max().item())
                cumulative = self._slice_data(torch.arange(0, max_idx + 1, device=self.data.device))
        else:
            current = TemporalGraphView(self.data, current_indices)
            cumulative = None
            if self.include_cumulative:
                max_idx = int(current_indices.max().item())
                cumulative = TemporalGraphView(self.data, slice(0, max_idx + 1))

        snapshot = TemporalGraphSnapshot(
            snapshot_id=self._current_snapshot_num,
            current=current,
            cumulative=cumulative
        )

        self._current_snapshot_num += 1
        return snapshot