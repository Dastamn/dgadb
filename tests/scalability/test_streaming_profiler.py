import pytest

from dgadb.scalability.streaming_profiler import StreamingProfiler


def test_profiler_records_snapshots_and_reports_throughput():
    profiler = StreamingProfiler(warmup_min=1, warmup_frac=0.0)
    # Simulate 4 snapshots, each scoring 100 edges, each taking 0.01 s.
    # Mean degree provided per snapshot.
    profiler.record(num_edges=100, mean_degree=2.5, elapsed_sec=0.01)
    profiler.record(num_edges=100, mean_degree=2.5, elapsed_sec=0.01)
    profiler.record(num_edges=100, mean_degree=2.5, elapsed_sec=0.01)
    profiler.record(num_edges=100, mean_degree=2.5, elapsed_sec=0.01)

    summary = profiler.summary()
    # Warmup discards the first snapshot (warmup_min=1), so 3 snapshots remain.
    assert summary["snapshots_after_warmup"] == 3
    assert summary["edges_after_warmup"] == 300
    assert summary["elapsed_after_warmup_sec"] == pytest.approx(0.03, rel=1e-6)
    assert summary["throughput_warmup_excluded"] == pytest.approx(10000.0, rel=1e-6)


def test_profiler_reports_latency_percentiles_and_ratio():
    profiler = StreamingProfiler(warmup_min=0, warmup_frac=0.0)
    # 100 snapshots with deterministic latencies 1..100 ms.
    for i in range(1, 101):
        profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=i / 1000.0)

    summary = profiler.summary()
    assert summary["latency_p50_ms"] == pytest.approx(50.5, rel=1e-3)
    assert summary["latency_p95_ms"] == pytest.approx(95.05, rel=1e-3)
    assert summary["latency_p99_ms"] == pytest.approx(99.01, rel=1e-3)
    assert summary["latency_mean_ms"] == pytest.approx(50.5, rel=1e-3)
    assert summary["latency_p99_over_p50"] == pytest.approx(99.01 / 50.5, rel=1e-3)


def test_profiler_flags_insufficient_batches_after_warmup():
    profiler = StreamingProfiler(warmup_min=5, warmup_frac=0.1, min_post_warmup=20)
    # Only 22 snapshots total, warmup discards 5, leaving 17 — below the floor of 20.
    for _ in range(22):
        profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=0.001)

    summary = profiler.summary()
    assert summary["insufficient_batches"] is True
    assert summary["snapshots_after_warmup"] == 17


def test_profiler_clears_insufficient_flag_when_enough_snapshots():
    profiler = StreamingProfiler(warmup_min=5, warmup_frac=0.1, min_post_warmup=20)
    for _ in range(30):
        profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=0.001)

    summary = profiler.summary()
    assert summary["insufficient_batches"] is False


def test_profiler_emits_sidecar_with_per_snapshot_arrays_and_windows():
    profiler = StreamingProfiler(warmup_min=0, warmup_frac=0.0, window_sec=1.0)
    # 5 snapshots: 0.4s + 0.4s + 0.4s + 0.4s + 0.4s (2.0s total), 100 edges each.
    # Window 1 (0.0..1.0s) contains snapshots ending at 0.4 and 0.8 -> 200 edges.
    # Window 2 (1.0..2.0s) contains snapshots ending at 1.2, 1.6, 2.0 -> 300 edges.
    for _ in range(5):
        profiler.record(num_edges=100, mean_degree=2.0, elapsed_sec=0.4)

    sidecar = profiler.to_sidecar()
    assert sidecar["per_snapshot_latency_ms"] == [400.0, 400.0, 400.0, 400.0, 400.0]
    assert sidecar["per_snapshot_edges"] == [100, 100, 100, 100, 100]
    assert sidecar["per_snapshot_mean_degree"] == [2.0, 2.0, 2.0, 2.0, 2.0]
    assert sidecar["windowed_throughput"] == [200.0, 300.0]


def test_profiler_omits_detection_delay_when_no_span_supplied():
    """When record() is never called with snapshot_span_sec, the
    detection-delay keys should be absent from the summary so that
    consumers can detect that the metric is not measured for this method."""
    profiler = StreamingProfiler(warmup_min=0, warmup_frac=0.0)
    for _ in range(5):
        profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=0.01)

    summary = profiler.summary()
    assert "detection_delay_mean_sec" not in summary
    assert "detection_delay_max_sec" not in summary
    assert "snapshots_with_span" not in summary


def test_profiler_reports_detection_delay_when_span_supplied():
    """Detection delay = buffering (span/2 for the typical edge, span for the
    worst) + scoring latency. With three snapshots of (span=10s, lat=1s),
    (span=20s, lat=2s), (span=30s, lat=3s) the typical-edge delays are
    (5+1)=6, (10+2)=12, (15+3)=18 — mean 12. The worst-edge delays are
    (10+1)=11, (20+2)=22, (30+3)=33 — max 33."""
    profiler = StreamingProfiler(warmup_min=0, warmup_frac=0.0)
    profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=1.0, snapshot_span_sec=10.0)
    profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=2.0, snapshot_span_sec=20.0)
    profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=3.0, snapshot_span_sec=30.0)

    summary = profiler.summary()
    assert summary["snapshots_with_span"] == 3
    assert summary["detection_delay_mean_sec"] == pytest.approx(12.0, rel=1e-6)
    assert summary["detection_delay_max_sec"] == pytest.approx(33.0, rel=1e-6)
    # With only 3 samples, p99 is the linearly-interpolated point at
    # k = (n-1) * 0.99 = 1.98 between sorted values [11, 22, 33], i.e.
    # 0.02 * 22 + 0.98 * 33 = 32.78. We just check it lies between the
    # second-largest worst-case delay and the largest.
    assert 22.0 < summary["detection_delay_p99_sec"] <= 33.0


def test_profiler_handles_mixed_span_and_no_span_records():
    """Snapshots that supply a span and snapshots that don't can coexist;
    detection-delay aggregates should be computed only from the spans
    that were supplied."""
    profiler = StreamingProfiler(warmup_min=0, warmup_frac=0.0)
    profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=1.0, snapshot_span_sec=10.0)
    profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=2.0)  # no span
    profiler.record(num_edges=10, mean_degree=1.0, elapsed_sec=3.0, snapshot_span_sec=30.0)

    summary = profiler.summary()
    assert summary["snapshots_with_span"] == 2
    # Mean of (5+1, 15+3) = mean(6, 18) = 12
    assert summary["detection_delay_mean_sec"] == pytest.approx(12.0, rel=1e-6)
    # Max of (10+1, 30+3) = 33
    assert summary["detection_delay_max_sec"] == pytest.approx(33.0, rel=1e-6)
