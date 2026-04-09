#!/usr/bin/env bash
#
# Dispatch the full Tier A scalability sweep: every (method, dataset) pair
# from DyGADBench, one run at a time, skipping cells that section 4.5 of
# the published paper already established as compute- or memory-bound.
#
# Each pair is launched as a separate `benchmark_scalability` invocation
# so that a failure in one cell (OOM, crash, missing data) does not cascade
# to the rest of the sweep. Per-pair logs land in
# $OUTPUT_DIR/logs/<method>__<dataset>.log; the script is idempotent — if a
# log file exists and its last line says SUCCESS, the pair is skipped on
# re-invocation.
#
# Usage:
#     ./scripts/run_scalability_sweep.sh [--epochs N] [--device cpu|gpu]
#                                        [--output-dir DIR]
#                                        [--methods "m1 m2 ..."]
#                                        [--datasets "d1 d2 ..."]
#                                        [--timeout SECONDS]
#                                        [--outer dataset|method]
#                                        [--pixi-env NAME]
#
# Defaults: --epochs 3, --device gpu, --output-dir benchmark-results/tier_a,
# --timeout 0 (no timeout), --outer dataset. Set a positive timeout to wrap
# each pair in `timeout <N>` so a hung run does not block the sweep; a pair
# killed by the timeout is recorded as FAILED with rc=124.
#
# --outer controls loop nesting. "dataset" (default) iterates every method
# on each dataset in turn, so every method finishes on the small datasets
# before any method starts on the large ones. "method" preserves the old
# method-outer order. Dataset-outer is strongly preferred under a time
# budget because a hang on (slow method, large dataset) does not block fast
# methods on other datasets.
#
# --pixi-env picks the pixi environment used to launch the benchmark. When
# unset, the script auto-selects "cuda" for --device gpu and "dev" for
# --device cpu, because the "dev" pixi feature installs CPU-only PyTorch
# wheels. Passing --device gpu from the "dev" env used to silently fall
# back to CPU; benchmark_scalability.py now raises instead, but
# auto-selecting the env here avoids the error in the first place.
#
# Success detection: a pair is only marked SUCCESS when (a) pixi exits 0,
# (b) the per-pair log contains no `FAILED:` / `Failed to load` line, and
# (c) the produced CSV row has an empty `error` column. This prevents
# silent load failures (which exit rc=0 with an error row) from being
# mistaken for success and skipped on re-runs.

set -euo pipefail

EPOCHS=3
DEVICE="gpu"
OUTPUT_DIR="benchmark-results/tier_a"
# Ordered fastest → slowest so cheap cells finish first and a hang on a
# slow method/dataset pair doesn't block the rest of the sweep.
# Methods: plain GNN baselines → temporal GNNs → heavier anomaly detectors.
# Datasets: sorted by edge count (see docs/dataset_edge_count_discrepancies.md).
METHODS="gcn gat graphsage addgraph rustgraph strgnn sad slade taddy generaldyg"
DATASETS="bitcoin-alpha email-dnc bitcoin-otc uc-social digg-homo as-topology enron epinions"
TIMEOUT_SEC=0
# Loop nesting order: "dataset" (default) iterates every method on each
# dataset in turn, so every method finishes on the small datasets before any
# method starts on the large ones. "method" iterates every dataset for each
# method, matching the historical behaviour. Dataset-outer is strongly
# preferred under a time budget because a hang on (slow method, large
# dataset) doesn't block fast methods on other datasets.
OUTER="dataset"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --methods)    METHODS="$2"; shift 2 ;;
        --datasets)   DATASETS="$2"; shift 2 ;;
        --timeout)    TIMEOUT_SEC="$2"; shift 2 ;;
        --outer)      OUTER="$2"; shift 2 ;;
        --pixi-env)   PIXI_ENV="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 2
            ;;
    esac
done

case "$OUTER" in
    dataset|method) ;;
    *)
        echo "ERROR: --outer must be 'dataset' or 'method', got '$OUTER'" >&2
        exit 2
        ;;
esac

# Pixi env selection. The `dev` env uses the CPU-only PyTorch wheel; the
# `cuda` env pulls in the CUDA-enabled wheel and the matching cuda runtime.
# Running `--device gpu` from `dev` silently falls through to CPU inside
# `torch.device("cuda" if torch.cuda.is_available() ...)` (benchmark_scalability.py
# now raises instead of falling back, but we resolve the env up front so
# operators don't hit that error every single pair).
# Override with --pixi-env if you have a non-standard env name.
if [[ -z "${PIXI_ENV:-}" ]]; then
    if [[ "$DEVICE" == "gpu" ]]; then
        PIXI_ENV="cuda"
    else
        PIXI_ENV="dev"
    fi
fi

# Require `timeout` (GNU coreutils) when the caller asked for one.
if [[ "$TIMEOUT_SEC" != "0" ]]; then
    if ! command -v timeout >/dev/null 2>&1; then
        echo "ERROR: --timeout requested but 'timeout' command not found" >&2
        echo "Install GNU coreutils (brew install coreutils on macOS)" >&2
        exit 2
    fi
fi

# Known-OOM cells inherited from section 4.5 of the published paper.
# Format: "method dataset reason".
declare -a KNOWN_OOM=(
    "taddy as-topology memory-bound"
    "taddy enron memory-bound"
    "taddy epinions memory-bound"
    "generaldyg as-topology memory-bound"
    "generaldyg epinions memory-bound"
    "strgnn epinions compute-bound"
    "sad epinions compute-bound"
    "addgraph epinions compute-bound"
)

is_known_oom() {
    local m="$1" d="$2"
    for entry in "${KNOWN_OOM[@]}"; do
        read -r om od _reason <<< "$entry"
        if [[ "$om" == "$m" && "$od" == "$d" ]]; then
            return 0
        fi
    done
    return 1
}

mkdir -p "$OUTPUT_DIR/logs"
SWEEP_LOG="$OUTPUT_DIR/logs/_sweep.log"
echo "=== sweep started at $(date --iso-8601=seconds 2>/dev/null || date) ===" >> "$SWEEP_LOG"
echo "methods: $METHODS" >> "$SWEEP_LOG"
echo "datasets: $DATASETS" >> "$SWEEP_LOG"
echo "epochs: $EPOCHS  device: $DEVICE  pixi_env: $PIXI_ENV  output: $OUTPUT_DIR  outer: $OUTER" >> "$SWEEP_LOG"
echo "[info] pixi env: $PIXI_ENV  device: $DEVICE"

total=0
completed=0
skipped_oom=0
skipped_done=0
failed=0

# Build the ordered list of (method, dataset) pairs in whichever loop nesting
# the caller requested. Emitted as "method dataset" lines for the inner loop
# to read with `read -r`.
pairs=()
if [[ "$OUTER" == "dataset" ]]; then
    for dataset in $DATASETS; do
        for method in $METHODS; do
            pairs+=("$method $dataset")
        done
    done
else
    for method in $METHODS; do
        for dataset in $DATASETS; do
            pairs+=("$method $dataset")
        done
    done
fi

for pair in "${pairs[@]}"; do
    read -r method dataset <<< "$pair"
    total=$((total + 1))
        log_file="$OUTPUT_DIR/logs/${method}__${dataset}.log"

        if is_known_oom "$method" "$dataset"; then
            echo "[SKIP oom] $method $dataset (known OOM from section 4.5)"
            echo "[SKIP oom] $method $dataset" >> "$SWEEP_LOG"
            skipped_oom=$((skipped_oom + 1))
            continue
        fi

        if [[ -f "$log_file" ]] && tail -n 1 "$log_file" 2>/dev/null | grep -q '^SUCCESS '; then
            echo "[SKIP done] $method $dataset (already completed)"
            skipped_done=$((skipped_done + 1))
            continue
        fi

        echo "[RUN ] $method $dataset"
        echo "[RUN ] $method $dataset  $(date --iso-8601=seconds 2>/dev/null || date)" >> "$SWEEP_LOG"

        # Build the command, optionally wrapped in timeout(1).
        # Using an array so quoting survives intact.
        cmd=(pixi run -e "$PIXI_ENV" python -m scripts.benchmark_scalability
             --methods "$method"
             --datasets "$dataset"
             --epochs "$EPOCHS"
             --device "$DEVICE"
             --output-dir "$OUTPUT_DIR")
        if [[ "$TIMEOUT_SEC" != "0" ]]; then
            cmd=(timeout --kill-after=30 "$TIMEOUT_SEC" "${cmd[@]}")
        fi

        rc=0
        "${cmd[@]}" > "$log_file" 2>&1 || rc=$?

        # Hardened success check: pixi rc must be 0, AND the log must not
        # contain a failure line, AND the benchmark must have written a
        # CSV row whose `error` column is empty.
        success=false
        reason=""
        if [[ $rc -ne 0 ]]; then
            if [[ "$TIMEOUT_SEC" != "0" && $rc -eq 124 ]]; then
                reason="timeout after ${TIMEOUT_SEC}s"
            else
                reason="rc=$rc"
            fi
        elif grep -qE '(FAILED:|Failed to load)' "$log_file"; then
            # Extract the first informative failure line for the summary.
            reason="benchmark error — $(grep -m1 -E '(FAILED:|Failed to load)' "$log_file" | tr -d '\n' | cut -c1-160)"
        else
            # Inspect the most recently written CSV for this pair's error column.
            # asdict writes the error column as the literal string "" when no error.
            error_cell=$(
                pixi run -e "$PIXI_ENV" python - <<'PY' 2>/dev/null "$OUTPUT_DIR" "$method" "$dataset"
import sys, glob, polars as pl
out_dir, method, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
paths = sorted(glob.glob(f"{out_dir}/benchmark_results_*.csv"))
for p in reversed(paths):
    try:
        df = pl.read_csv(p, infer_schema_length=10000)
    except Exception:
        continue
    rows = df.filter((pl.col("method") == method) & (pl.col("dataset") == dataset))
    if rows.height == 0:
        continue
    err = rows.row(rows.height - 1, named=True).get("error")
    if err is None or err == "" or str(err).lower() == "null":
        print("OK")
    else:
        print(f"ERR:{err}")
    break
else:
    print("NOCSV")
PY
            )
            case "$error_cell" in
                OK)     success=true ;;
                NOCSV)  reason="no CSV row produced" ;;
                ERR:*)  reason="csv error column: ${error_cell#ERR:}" ;;
                *)      reason="csv inspection inconclusive: $error_cell" ;;
            esac
        fi

        if $success; then
            echo "SUCCESS $method $dataset" >> "$log_file"
            completed=$((completed + 1))
            echo "[ OK ] $method $dataset"
        else
            echo "FAILED $method $dataset ($reason)" >> "$log_file"
            failed=$((failed + 1))
            echo "[FAIL] $method $dataset ($reason — see $log_file)"
            echo "[FAIL] $method $dataset $reason" >> "$SWEEP_LOG"
        fi
done

echo
echo "=== sweep summary ==="
echo "  total pairs:         $total"
echo "  completed:           $completed"
echo "  skipped (done):      $skipped_done"
echo "  skipped (known OOM): $skipped_oom"
echo "  failed:              $failed"
echo

{
    echo "=== sweep finished at $(date --iso-8601=seconds 2>/dev/null || date) ==="
    echo "completed=$completed skipped_done=$skipped_done skipped_oom=$skipped_oom failed=$failed"
} >> "$SWEEP_LOG"

if [[ $failed -gt 0 ]]; then
    exit 1
fi
