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
        min_post_warmup: Minimum number of snapshots required after warmup
            for the throughput / latency measurement to be considered
            reliable. ``summary()`` reports ``insufficient_batches=True``
            when this floor is not met.
        window_sec: Width of the throughput window used by ``to_sidecar``,
            in seconds. Each snapshot's edges are attributed to the window
            in which the snapshot ends.
    """

    warmup_min: int = 5
    warmup_frac: float = 0.1
    min_post_warmup: int = 20
    window_sec: float = 1.0
    _edges: list[int] = field(default_factory=list)
    _mean_degrees: list[float] = field(default_factory=list)
    _elapsed: list[float] = field(default_factory=list)

    def record(self, *, num_edges: int, mean_degree: float, elapsed_sec: float) -> None:
        """Record one snapshot's inference timing.

        Args:
            num_edges: Number of edges scored in this snapshot.
            mean_degree: Mean degree of the snapshot's edges.
            elapsed_sec: Wall-clock elapsed time, in seconds, for scoring this snapshot.
        """
        self._edges.append(int(num_edges))
        self._mean_degrees.append(float(mean_degree))
        self._elapsed.append(float(elapsed_sec))

    def _warmup_count(self) -> int:
        """Return the number of leading snapshots to discard as warmup.

        Effective warmup is ``min(n, max(warmup_min, ceil(warmup_frac * n)))``,
        where ``n`` is the total number of recorded snapshots. The outer
        ``min(n, ...)`` ensures the warmup never exceeds the total snapshot
        count when very few snapshots have been recorded.
        """
        from math import ceil

        n = len(self._elapsed)
        return min(n, max(self.warmup_min, ceil(self.warmup_frac * n)))

    def summary(self) -> dict[str, Any]:
        """Return warmup-excluded streaming metrics as a flat dict.

        Keys (all measured after the warmup window has been discarded):
            - ``warmup_snapshots`` (int): number of snapshots discarded.
            - ``snapshots_after_warmup`` (int): number of snapshots measured.
            - ``edges_after_warmup`` (int): total edges scored.
            - ``elapsed_after_warmup_sec`` (float): total wall-clock seconds.
            - ``throughput_warmup_excluded`` (float): edges scored per second.
            - ``latency_mean_ms`` (float): mean per-snapshot latency, ms.
            - ``latency_p50_ms`` (float): 50th percentile (median) latency, ms.
            - ``latency_p95_ms`` (float): 95th percentile latency, ms.
            - ``latency_p99_ms`` (float): 99th percentile latency, ms.
            - ``latency_p99_over_p50`` (float): tail-to-median ratio, a single
              number describing how predictable the per-snapshot cost is.
            - ``insufficient_batches`` (bool): True when the post-warmup
              snapshot count is below ``min_post_warmup``; the throughput
              and latency numbers should be treated with caution.
        """
        import statistics

        w = self._warmup_count()
        edges = self._edges[w:]
        elapsed = self._elapsed[w:]
        n_after = len(elapsed)
        total_edges = sum(edges)
        total_elapsed = sum(elapsed)
        throughput = total_edges / total_elapsed if total_elapsed > 0 else 0.0

        latencies_ms = sorted(s * 1000.0 for s in elapsed)

        def pct(p: float) -> float:
            if not latencies_ms:
                return 0.0
            # Linear interpolation between order statistics.
            k = (len(latencies_ms) - 1) * p
            lo = int(k)
            hi = min(lo + 1, len(latencies_ms) - 1)
            frac = k - lo
            return latencies_ms[lo] * (1 - frac) + latencies_ms[hi] * frac

        p50 = pct(0.50)
        p95 = pct(0.95)
        p99 = pct(0.99)
        mean = statistics.fmean(latencies_ms) if latencies_ms else 0.0
        ratio = (p99 / p50) if p50 > 0 else 0.0

        return {
            "warmup_snapshots": w,
            "snapshots_after_warmup": n_after,
            "edges_after_warmup": total_edges,
            "elapsed_after_warmup_sec": total_elapsed,
            "throughput_warmup_excluded": throughput,
            "latency_mean_ms": mean,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "latency_p99_over_p50": ratio,
            "insufficient_batches": n_after < self.min_post_warmup,
        }

    def to_sidecar(self) -> dict[str, Any]:
        """Return per-snapshot arrays and a windowed throughput series.

        Windowed throughput is computed by walking the snapshot timeline
        (each snapshot's edges are attributed to the window in which it
        ends) and dividing edges-in-window by ``window_sec``. The result
        is intended to be persisted as a JSON sidecar alongside the main
        benchmark results, so the appendix figures can be regenerated
        without re-running inference.

        Keys:
            - ``per_snapshot_latency_ms`` (list[float]): every recorded
              snapshot's latency in milliseconds, in arrival order.
            - ``per_snapshot_edges`` (list[int]): edge count per snapshot.
            - ``per_snapshot_mean_degree`` (list[float]): mean edge degree
              per snapshot, in the same order.
            - ``windowed_throughput`` (list[float]): edges scored per
              ``window_sec``-wide bucket, in chronological order.
        """
        latencies_ms = [s * 1000.0 for s in self._elapsed]

        from math import ceil

        windows: list[float] = []
        edges_in_window = 0
        cumulative = 0.0
        window_idx = 0
        for edges, dur in zip(self._edges, self._elapsed):
            cumulative += dur
            # Snapshot belongs to window ceil(cumulative / window_sec) - 1,
            # so a snapshot ending exactly at the boundary (e.g. 1.0) belongs
            # to window 0 (0..1], not window 1 (1..2].
            target_window = ceil(cumulative / self.window_sec) - 1
            while window_idx < target_window:
                windows.append(edges_in_window / self.window_sec)
                edges_in_window = 0
                window_idx += 1
            edges_in_window += edges
        if edges_in_window > 0:
            windows.append(edges_in_window / self.window_sec)

        return {
            "per_snapshot_latency_ms": latencies_ms,
            "per_snapshot_edges": list(self._edges),
            "per_snapshot_mean_degree": list(self._mean_degrees),
            "windowed_throughput": windows,
        }
