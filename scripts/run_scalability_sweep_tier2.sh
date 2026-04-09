#!/usr/bin/env bash
#
# Tier 2 scalability sweep: epinions only, for every method that is not
# already known to OOM or be compute-bound on it (see the KNOWN_OOM list in
# run_scalability_sweep.sh).
#
# Run this AFTER tier 1 completes. Epinions is substantially larger than
# the other datasets and is the stretch goal of the sweep. Several methods
# are inherited as known-OOM from section 4.5 of the paper and are skipped
# automatically.
#
# This wrapper pins a 3-hour per-pair timeout, which is long enough for
# GCN/GAT/GraphSAGE with proper threading and short enough to bound the
# total wall-time. With --kill-after=30 the timeout is hard.
#
#     export OMP_NUM_THREADS=$(nproc)
#     export MKL_NUM_THREADS=$(nproc)
#     ./scripts/run_scalability_sweep_tier2.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TIER2_DATASETS="epinions"
DEFAULT_TIMEOUT=10800   # 3 hours per pair

"$HERE/run_scalability_sweep.sh" \
    --datasets "$TIER2_DATASETS" \
    --outer dataset \
    --timeout "$DEFAULT_TIMEOUT" \
    --output-dir benchmark-results/tier_a \
    "$@"
