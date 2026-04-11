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
    _spans_sec: list[float] = field(default_factory=list)

    def record(
        self,
        *,
        num_edges: int,
        mean_degree: float,
        elapsed_sec: float,
        snapshot_span_sec: float | None = None,
    ) -> None:
        """Record one snapshot's inference timing.

        Args:
            num_edges: Number of edges scored in this snapshot.
            mean_degree: Mean degree of the snapshot's edges.
            elapsed_sec: Wall-clock elapsed time, in seconds, for scoring this snapshot.
            snapshot_span_sec: Optional data-time span of the snapshot, in
                seconds — i.e. ``max(edge_timestamps) - min(edge_timestamps)``.
                When supplied, the profiler reports detection-delay aggregates
                in :py:meth:`summary` that combine the snapshot's buffering
                delay (the time edges spent waiting in the snapshot buffer
                before scoring could start) with the scoring latency. Pass
                ``None`` for methods whose inference loop does not iterate
                over snapshots in the natural way (e.g. methods that batch
                all snapshots into a single forward pass): detection-delay
                stats will simply be omitted from the summary in that case.
        """
        self._edges.append(int(num_edges))
        self._mean_degrees.append(float(mean_degree))
        self._elapsed.append(float(elapsed_sec))
        self._spans_sec.append(float(snapshot_span_sec) if snapshot_span_sec is not None else float("nan"))

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
            - ``latency_mean_ms`` (float): mean per-snapshot scoring latency, ms.
            - ``latency_p50_ms`` (float): 50th percentile (median) scoring latency, ms.
            - ``latency_p95_ms`` (float): 95th percentile scoring latency, ms.
            - ``latency_p99_ms`` (float): 99th percentile scoring latency, ms.
            - ``latency_p99_over_p50`` (float): tail-to-median ratio, a single
              number describing how predictable the per-snapshot cost is.
            - ``insufficient_batches`` (bool): True when the post-warmup
              snapshot count is below ``min_post_warmup``; the throughput
              and latency numbers should be treated with caution.

        When ``snapshot_span_sec`` was supplied to ``record`` for at least
        one post-warmup snapshot, the following detection-delay metrics are
        also reported. They model an edge's end-to-end delay as the sum of
        (a) buffering delay — the time the edge spent in the snapshot buffer
        before scoring could start, approximated from the snapshot's
        data-time span — and (b) scoring latency:

            - ``detection_delay_mean_sec`` (float): mean over snapshots of
              ``span/2 + scoring_latency`` (i.e. the typical edge in a
              uniformly-filled snapshot).
            - ``detection_delay_max_sec`` (float): max over snapshots of
              ``span + scoring_latency`` (i.e. the worst edge — the oldest
              one in the snapshot).
            - ``detection_delay_p95_sec`` / ``detection_delay_p99_sec`` (float):
              tail percentiles of the worst-case-per-snapshot delay.
            - ``snapshots_with_span`` (int): number of post-warmup snapshots
              for which a span was recorded.

        These keys are absent when no spans were ever supplied (e.g. methods
        whose inference loop batches across snapshots and cannot expose a
        per-snapshot span).
        """
        import math
        import statistics

        w = self._warmup_count()
        edges = self._edges[w:]
        elapsed = self._elapsed[w:]
        spans = self._spans_sec[w:]
        n_after = len(elapsed)
        total_edges = sum(edges)
        total_elapsed = sum(elapsed)
        throughput = total_edges / total_elapsed if total_elapsed > 0 else 0.0

        latencies_ms = sorted(s * 1000.0 for s in elapsed)

        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            # Linear interpolation between order statistics.
            k = (len(values) - 1) * p
            lo = int(k)
            hi = min(lo + 1, len(values) - 1)
            frac = k - lo
            return values[lo] * (1 - frac) + values[hi] * frac

        p50 = pct(latencies_ms, 0.50)
        p95 = pct(latencies_ms, 0.95)
        p99 = pct(latencies_ms, 0.99)
        mean = statistics.fmean(latencies_ms) if latencies_ms else 0.0
        ratio = (p99 / p50) if p50 > 0 else 0.0

        out: dict[str, Any] = {
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

        # Detection delay: only report when at least one post-warmup snapshot
        # supplied a span. We model the per-snapshot delay as buffering +
        # scoring; the buffering delay for the typical edge in a snapshot of
        # data-time span W is W/2 (assuming uniformly distributed arrival
        # within the snapshot), and W for the oldest edge.
        valid_pairs = [
            (span, lat)
            for span, lat in zip(spans, elapsed)
            if not math.isnan(span)
        ]
        if valid_pairs:
            mean_delays = [span / 2.0 + lat for span, lat in valid_pairs]
            max_delays = sorted(span + lat for span, lat in valid_pairs)
            spans_only = [span for span, _ in valid_pairs]
            out["snapshots_with_span"] = len(valid_pairs)
            out["detection_delay_mean_sec"] = statistics.fmean(mean_delays)
            out["detection_delay_max_sec"] = max_delays[-1] if max_delays else 0.0
            out["detection_delay_p95_sec"] = pct(max_delays, 0.95)
            out["detection_delay_p99_sec"] = pct(max_delays, 0.99)
            # Also surface the snapshot data-time span aggregates so consumers
            # can decompose detection delay into its two components: buffering
            # delay (≈ span/2) and scoring latency (already reported above).
            # The buffering component is a function of the snapshot window
            # size, which is a benchmark-tuning parameter rather than a
            # property of the method.
            out["snapshot_span_mean_sec"] = statistics.fmean(spans_only)
            out["snapshot_span_max_sec"] = max(spans_only)

        return out

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
            "per_snapshot_span_sec": list(self._spans_sec),
            "windowed_throughput": windows,
        }
