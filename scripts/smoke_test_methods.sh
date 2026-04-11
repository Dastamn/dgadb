#!/usr/bin/env bash
#
# smoke_test_methods.sh — per-method end-to-end smoke test
#
# Runs `dgadb run` once for every method on a small dataset with a minimal
# anomaly configuration (1 epoch, 1 anomaly type, 1 rate, 1 duration). The
# goal is to catch any method that breaks at import, setup, training, or
# inference time after a refactor or merge. It exercises:
#
#   - Model constructor and `.setup(data)`
#   - `.train()` loop (at least one epoch, train + val loader)
#   - `ExperimentRunner.evaluate()` which calls `.run_inference()` on the
#     test split
#   - AimCallback logging
#   - ResourceMonitor callback
#
# This is COMPLEMENTARY to `run_scalability_sweep.sh`, which covers the
# `dgadb scalability` path (StreamingProfiler + warmup-excluded metrics).
# For full per-method coverage after a big merge, run both scripts.
#
# Usage:
#     scripts/smoke_test_methods.sh                       # cpu, bitcoin-alpha
#     scripts/smoke_test_methods.sh --device gpu          # gpu (cuda env)
#     scripts/smoke_test_methods.sh --dataset bitcoin-otc
#     scripts/smoke_test_methods.sh --methods "gcn gat"   # subset
#     scripts/smoke_test_methods.sh --epochs 2 --timeout 900
#
# Success detection:
#     `dgadb run` does NOT propagate exceptions from its process pool —
#     ProcessPoolExecutor re-raises the subprocess exception in the main
#     process, which catches it with `logging.error("Experiment failed:")`
#     and exits 0. That main-process log line is itself silently dropped
#     because runner.py only calls logging.basicConfig when invoked as
#     __main__ (i.e. `python -m dgadb.experiment.runner`), not when
#     invoked via `dgadb run`. So we can't rely on "Experiment failed:"
#     appearing.
#
#     The subprocess, however, configures its own logger via
#     `setup_worker_logging` and emits "ExperimentRunner: Done." as the
#     last line of a successful run. This script treats a method as
#     passing only when (a) pixi exits 0, (b) the log has
#     "ExperimentRunner: Done.", AND (c) the log has no Python traceback.
#
# Exit code: 0 if every method passes, 1 if any failed or timed out, 2 on
# argument errors.

set -euo pipefail

# Defaults — tuned for a fast smoke (10 methods, 1 epoch, 1 anomaly config).
METHODS="sad taddy slade strgnn rustgraph gcn gat graphsage generaldyg addgraph"
DATASET="bitcoin-alpha"
EPOCHS=1
ANOM_TYPE="random"
ANOM_RATE="0.05"
ANOM_DURATION="medium"
DEVICE="cpu"
OUTPUT_DIR="benchmark-results/smoke_methods"
TIMEOUT_SEC=600
EXPERIMENT_NAME="smoke_methods"
PIXI_ENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)         METHODS="$2"; shift 2 ;;
        --dataset)         DATASET="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --anom-type)       ANOM_TYPE="$2"; shift 2 ;;
        --anom-rate)       ANOM_RATE="$2"; shift 2 ;;
        --anom-duration)   ANOM_DURATION="$2"; shift 2 ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --timeout)         TIMEOUT_SEC="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --pixi-env)        PIXI_ENV="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,45p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# Auto-pick pixi env based on device (same policy as run_scalability_sweep.sh):
# `dev` ships CPU-only PyTorch wheels; `cuda` has the GPU-enabled wheel. If
# you have a non-standard env name, override with --pixi-env.
if [[ -z "$PIXI_ENV" ]]; then
    if [[ "$DEVICE" == "gpu" ]]; then
        PIXI_ENV="cuda"
    else
        PIXI_ENV="dev"
    fi
fi

# `timeout` (GNU coreutils) is required when a non-zero TIMEOUT_SEC is set.
if [[ "$TIMEOUT_SEC" != "0" ]]; then
    if ! command -v timeout >/dev/null 2>&1; then
        echo "ERROR: --timeout requested but 'timeout' command not found" >&2
        echo "Install GNU coreutils (brew install coreutils on macOS)" >&2
        exit 2
    fi
fi

# Per-run output directory, timestamped and device-tagged so CPU and GPU
# runs don't overwrite each other when launched from the same checkout.
timestamp=$(date +%Y%m%d_%H%M%S)
run_dir="$OUTPUT_DIR/${timestamp}_${DEVICE}"
mkdir -p "$run_dir"
sweep_log="$run_dir/_smoke.log"

# Machine-readable per-method results live here. Tab-separated columns:
#   method<TAB>result<TAB>elapsed_sec<TAB>reason
# A row is appended by each method iteration below. We use a file rather
# than bash associative arrays because this script must run on macOS's
# default bash 3.2 (no `declare -A`) as well as Linux.
results_tsv="$run_dir/_results.tsv"
printf "method\tresult\telapsed_sec\treason\n" > "$results_tsv"

{
    echo "=== smoke_test_methods started at $(date --iso-8601=seconds 2>/dev/null || date) ==="
    echo "methods:         $METHODS"
    echo "dataset:         $DATASET"
    echo "epochs:          $EPOCHS"
    echo "anomaly:         type=$ANOM_TYPE  rate=$ANOM_RATE  duration=$ANOM_DURATION"
    echo "device:          $DEVICE  (pixi env: $PIXI_ENV)"
    echo "timeout:         ${TIMEOUT_SEC}s per method"
    echo "experiment name: $EXPERIMENT_NAME"
    echo "output:          $run_dir"
    echo "========================================================"
} | tee "$sweep_log"

for method in $METHODS; do
    log_file="$run_dir/${method}.log"
    echo
    echo "[RUN ] $method"
    echo "[RUN ] $method  $(date --iso-8601=seconds 2>/dev/null || date)" >> "$sweep_log"

    start_s=$(date +%s)

    # Build the command as an array so quoting survives intact.
    cmd=(pixi run -e "$PIXI_ENV" dgadb run
         --method "$method"
         --datasets "$DATASET"
         --epochs "$EPOCHS"
         --anom-types "$ANOM_TYPE"
         --anom-rates "$ANOM_RATE"
         --anom-durations "$ANOM_DURATION"
         --concurrency 1
         --device "$DEVICE"
         --experiment-name "$EXPERIMENT_NAME")
    if [[ "$TIMEOUT_SEC" != "0" ]]; then
        cmd=(timeout --kill-after=30 "$TIMEOUT_SEC" "${cmd[@]}")
    fi

    rc=0
    "${cmd[@]}" > "$log_file" 2>&1 || rc=$?
    elapsed=$(( $(date +%s) - start_s ))

    # Success detection. A successful run ends with the subprocess
    # logging "ExperimentRunner: Done." from ExperimentRunner.run(). A
    # subprocess crash leaves a Python traceback in the captured output
    # (stderr of the spawn process) and no "Done." line. See header
    # comment for why we cannot rely on "Experiment failed:" here.
    result="FAIL"
    reason=""
    if [[ $rc -ne 0 ]]; then
        if [[ "$TIMEOUT_SEC" != "0" && $rc -eq 124 ]]; then
            reason="timeout after ${TIMEOUT_SEC}s"
        else
            reason="pixi rc=$rc"
        fi
    elif grep -q "Traceback (most recent call last)" "$log_file"; then
        # Grab the line immediately after the last traceback — that is
        # the exception type + message — and truncate for the table.
        first_err=$(grep -A 20 "Traceback (most recent call last)" "$log_file" | grep -m1 -E "^[A-Z][A-Za-z]*Error|^[A-Z][A-Za-z]*Exception" | tr -d '\n' | cut -c1-200)
        if [[ -z "$first_err" ]]; then
            first_err="see traceback in log"
        fi
        reason="subprocess raised — $first_err"
    elif ! grep -q "ExperimentRunner: Done\." "$log_file"; then
        reason="no 'ExperimentRunner: Done.' marker — run did not reach completion"
    else
        result="PASS"
    fi

    # Append one row to the results TSV. Tabs are stripped from `reason`
    # so the TSV stays parseable even if grep picked up a message with
    # embedded tabs.
    sanitized_reason=$(printf '%s' "$reason" | tr '\t' ' ')
    printf "%s\t%s\t%s\t%s\n" "$method" "$result" "$elapsed" "$sanitized_reason" >> "$results_tsv"

    if [[ "$result" == "PASS" ]]; then
        echo "[ OK ] $method  (${elapsed}s)"
        echo "[ OK ] $method  (${elapsed}s)" >> "$sweep_log"
    else
        echo "[FAIL] $method  (${elapsed}s — $reason)"
        echo "[FAIL] $method  (${elapsed}s — $reason — see $log_file)" >> "$sweep_log"
    fi
done

# Summary table, rendered from the TSV.
echo
echo "=== smoke_test_methods summary ==="
echo "=== smoke_test_methods summary ===" >> "$sweep_log"
printf "%-12s %-6s %8s  %s\n" "method" "result" "time" "reason"
printf "%-12s %-6s %8s  %s\n" "method" "result" "time" "reason" >> "$sweep_log"
printf "%-12s %-6s %8s  %s\n" "------------" "------" "--------" "------"
printf "%-12s %-6s %8s  %s\n" "------------" "------" "--------" "------" >> "$sweep_log"

pass=0
fail=0
# Skip the header row (tail -n +2).
while IFS=$'\t' read -r method result elapsed reason; do
    printf "%-12s %-6s %8s  %s\n" "$method" "$result" "${elapsed}s" "$reason"
    printf "%-12s %-6s %8s  %s\n" "$method" "$result" "${elapsed}s" "$reason" >> "$sweep_log"
    if [[ "$result" == "PASS" ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
    fi
done < <(tail -n +2 "$results_tsv")

echo
echo "  passed:      $pass"
echo "  failed:      $fail"
echo "  logs:        $run_dir"
echo "  results tsv: $results_tsv"
{
    echo
    echo "  passed:      $pass"
    echo "  failed:      $fail"
    echo "  logs:        $run_dir"
    echo "  results tsv: $results_tsv"
    echo "=== smoke_test_methods finished at $(date --iso-8601=seconds 2>/dev/null || date) ==="
} >> "$sweep_log"

if [[ $fail -gt 0 ]]; then
    exit 1
fi
exit 0
