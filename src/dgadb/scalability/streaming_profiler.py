"""Streaming-scalability profiler for inference loops.

Records per-snapshot wall-clock latency, edge counts, and mean degree, and
reports warmup-excluded throughput plus latency distribution statistics.
The unit of measurement is the *snapshot* (not the individual edge), because
DGAD methods score a snapshot in a single forward pass — per-edge latency
under that model is just batch_time / batch_size = 1 / throughput and
carries no extra information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamingProfiler:
    """Records per-snapshot inference timings and computes streaming metrics.

    Attributes:
        warmup_min: Minimum number of snapshots to discard as warmup.
        warmup_frac: Additional fraction of total snapshots to discard
            (the effective warmup is ``max(warmup_min, ceil(warmup_frac * N))``).
    """

    warmup_min: int = 5
    warmup_frac: float = 0.1
    _edges: list[int] = field(default_factory=list)
    _mean_degrees: list[float] = field(default_factory=list)
    _elapsed: list[float] = field(default_factory=list)

    def record(self, *, num_edges: int, mean_degree: float, elapsed_sec: float) -> None:
        """Append one snapshot's measurements."""
        self._edges.append(int(num_edges))
        self._mean_degrees.append(float(mean_degree))
        self._elapsed.append(float(elapsed_sec))

    def _warmup_count(self) -> int:
        from math import ceil

        n = len(self._elapsed)
        return min(n, max(self.warmup_min, ceil(self.warmup_frac * n)))

    def summary(self) -> dict[str, Any]:
        """Return a flat dict of warmup-excluded streaming metrics."""
        w = self._warmup_count()
        edges = self._edges[w:]
        elapsed = self._elapsed[w:]
        n_after = len(elapsed)
        total_edges = sum(edges)
        total_elapsed = sum(elapsed)
        throughput = total_edges / total_elapsed if total_elapsed > 0 else 0.0
        return {
            "warmup_snapshots": w,
            "snapshots_after_warmup": n_after,
            "edges_after_warmup": total_edges,
            "elapsed_after_warmup_sec": total_elapsed,
            "throughput_warmup_excluded": throughput,
        }
