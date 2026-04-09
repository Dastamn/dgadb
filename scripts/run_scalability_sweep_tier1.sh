#!/usr/bin/env bash
#
# Tier 1 scalability sweep: every (method, dataset) pair from DyGADBench
# EXCEPT epinions. Use this first under any time budget, because epinions
# is substantially larger than the other datasets and the compute-heavy
# methods are hours-to-intractable on it; running epinions last guarantees
# a complete table for every other dataset even if the overall sweep is
# interrupted.
#
# This is a thin wrapper around run_scalability_sweep.sh. It pins:
#   - dataset-outer ordering (every method finishes on bitcoin-alpha before
#     anyone starts on email-dnc, etc.), so a slow method on a medium
#     dataset doesn't block fast methods on other datasets.
#   - a 30-minute per-pair timeout so an unexpectedly slow cell is recorded
#     as FAILED rc=124 instead of consuming the whole budget.
#
# All other flags (epochs, device, output-dir, methods subset) pass through
# to the underlying dispatcher. The OMP thread count should be exported in
# the calling shell so PyTorch picks it up at import time:
#
#     export OMP_NUM_THREADS=$(nproc)
#     export MKL_NUM_THREADS=$(nproc)
#     ./scripts/run_scalability_sweep_tier1.sh
#
# benchmark_scalability.py also calls torch.set_num_threads() from these
# same variables, so the override lands even if PyTorch's MKL build defaults
# to 1 thread.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TIER1_DATASETS="bitcoin-alpha email-dnc bitcoin-otc uc-social digg-homo as-topology enron"
DEFAULT_TIMEOUT=1800   # 30 minutes per pair

"$HERE/run_scalability_sweep.sh" \
    --datasets "$TIER1_DATASETS" \
    --outer dataset \
    --timeout "$DEFAULT_TIMEOUT" \
    --output-dir benchmark-results/tier_a \
    "$@"
