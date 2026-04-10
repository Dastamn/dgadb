#!/usr/bin/env bash
#
# Tier 2 scalability sweep: DGraph only, for every method that is not
# already known to OOM or be compute-bound on it (see the KNOWN_OOM list in
# run_scalability_sweep.sh).
#
# Run this AFTER tier 1 completes. DGraph (3.7M nodes, 4.3M edges) is
# substantially larger than the other datasets and is the stretch goal
# of the sweep. Most non-baseline methods are expected to OOM and are
# skipped automatically; the primary purpose is to characterize the
# GNN baselines (GCN, GAT, GraphSAGE) and SLADE at large scale.
#
# This wrapper pins a 3-hour per-pair timeout, which is long enough for
# the baselines and short enough to bound total wall-time.
#
#     export OMP_NUM_THREADS=$(nproc)
#     export MKL_NUM_THREADS=$(nproc)
#     ./scripts/run_scalability_sweep_tier2.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TIER2_DATASETS="dgraph"
DEFAULT_TIMEOUT=10800   # 3 hours per pair

"$HERE/run_scalability_sweep.sh" \
    --datasets "$TIER2_DATASETS" \
    --outer dataset \
    --timeout "$DEFAULT_TIMEOUT" \
    --output-dir benchmark-results/tier_a \
    "$@"
