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
