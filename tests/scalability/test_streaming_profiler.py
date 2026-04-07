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
