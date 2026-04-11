#!/usr/bin/env bash
#
# smoke_test.sh — two-phase end-to-end smoke test
#
# Covers both axes of the method × dataset matrix without running the
# full cartesian product:
#
#   Phase 1 (methods axis): every method in $METHODS against the single
#     baseline dataset $PHASE1_DATASET. Catches methods that break at
#     import, setup, training, or inference time after a refactor or
#     merge. Default: 10 methods × bitcoin-alpha.
#
#   Phase 2 (datasets axis): the single baseline method $PHASE2_METHOD
#     against every dataset in $DATASETS (skipping $PHASE1_DATASET to
#     avoid redundancy). Catches datasets that break at load/
#     preprocessing/splitting time. Default: gcn × everything else.
#
# The union gives ~1 cell per method + ~1 cell per dataset, so every
# method is exercised once and every dataset is exercised once, for a
# total of roughly (num_methods + num_datasets - 1) cells instead of
# the cartesian (num_methods × num_datasets).
#
# Each cell exercises:
#
#   - Model constructor and `.setup(data)`
#   - `.train()` loop (at least one epoch, train + val loader)
#   - `ExperimentRunner.evaluate()` which calls `.run_inference()` on
#     the test split
#   - AimCallback logging
#   - ResourceMonitor callback
#
# This is COMPLEMENTARY to `run_scalability_sweep.sh`, which covers the
# `dgadb scalability` path (StreamingProfiler + warmup-excluded metrics).
# For full coverage after a big merge, run both scripts.
#
# Usage:
#     scripts/smoke_test.sh                                # cpu, full default
#     scripts/smoke_test.sh --device gpu                   # gpu (cuda env)
#     scripts/smoke_test.sh --methods "gcn gat" --skip-phase2
#     scripts/smoke_test.sh --datasets "bitcoin-alpha bitcoin-otc" --skip-phase1
#     scripts/smoke_test.sh --epochs 2 --timeout 900 --device gpu
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
#     last line of a successful run. This script treats a cell as
#     passing only when (a) pixi exits 0, (b) the log has
#     "ExperimentRunner: Done.", AND (c) the log has no Python traceback.
#
# Exit code: 0 if every cell passes, 1 if any failed or timed out, 2 on
# argument errors.

set -euo pipefail

# Defaults — tuned for a fast smoke.
METHODS="sad taddy slade strgnn rustgraph gcn gat graphsage generaldyg addgraph"
# Phase 2 datasets. This is the list of datasets exercised via
# $PHASE2_METHOD. Includes every dataset config we ship; override with
# --datasets to narrow it down on constrained hardware or when some
# datasets are not yet downloaded.
DATASETS="bitcoin-alpha bitcoin-otc email-dnc uc-social digg-homo as-topology mooc wiki reddit epinions enron dgraph"
PHASE1_DATASET="bitcoin-alpha"
PHASE2_METHOD="gcn"

EPOCHS=1
ANOM_TYPE="random"
ANOM_RATE="0.05"
ANOM_DURATION="medium"
DEVICE="cpu"
OUTPUT_DIR="benchmark-results/smoke"
TIMEOUT_SEC=600
EXPERIMENT_NAME="smoke"
PIXI_ENV=""
SKIP_PHASE1=false
SKIP_PHASE2=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)         METHODS="$2"; shift 2 ;;
        --datasets)        DATASETS="$2"; shift 2 ;;
        --phase1-dataset)  PHASE1_DATASET="$2"; shift 2 ;;
        --phase2-method)   PHASE2_METHOD="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --anom-type)       ANOM_TYPE="$2"; shift 2 ;;
        --anom-rate)       ANOM_RATE="$2"; shift 2 ;;
        --anom-duration)   ANOM_DURATION="$2"; shift 2 ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --timeout)         TIMEOUT_SEC="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --pixi-env)        PIXI_ENV="$2"; shift 2 ;;
        --skip-phase1)     SKIP_PHASE1=true; shift ;;
        --skip-phase2)     SKIP_PHASE2=true; shift ;;
        -h|--help)
            sed -n '3,60p' "$0"
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

# Machine-readable per-cell results. Tab-separated columns:
#   phase<TAB>method<TAB>dataset<TAB>result<TAB>elapsed_sec<TAB>reason
# We use a file rather than bash associative arrays because this script
# must run on macOS's default bash 3.2 (no `declare -A`) as well as Linux.
results_tsv="$run_dir/_results.tsv"
printf "phase\tmethod\tdataset\tresult\telapsed_sec\treason\n" > "$results_tsv"

{
    echo "=== smoke_test started at $(date --iso-8601=seconds 2>/dev/null || date) ==="
    echo "phase 1:          $([ "$SKIP_PHASE1" = true ] && echo skipped || echo "methods × $PHASE1_DATASET")"
    echo "  methods:        $METHODS"
    echo "phase 2:          $([ "$SKIP_PHASE2" = true ] && echo skipped || echo "$PHASE2_METHOD × datasets")"
    echo "  datasets:       $DATASETS"
    echo "epochs:           $EPOCHS"
    echo "anomaly:          type=$ANOM_TYPE  rate=$ANOM_RATE  duration=$ANOM_DURATION"
    echo "device:           $DEVICE  (pixi env: $PIXI_ENV)"
    echo "timeout:          ${TIMEOUT_SEC}s per cell"
    echo "experiment name:  $EXPERIMENT_NAME"
    echo "output:           $run_dir"
    echo "========================================================"
} | tee "$sweep_log"

# Return 0 if $1 appears as a whitespace-separated token in $2. Used
# for the phase-2 dedup check below.
contains_word() {
    local needle="$1"
    local haystack="$2"
    local word
    for word in $haystack; do
        if [[ "$word" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

# Shared per-cell runner. Arguments: phase, method, dataset.
# Appends one row to $results_tsv, prints a status line, and writes a
# per-cell log to $run_dir/<method>__<dataset>.log. Reads the sweep-wide
# config from the enclosing script's globals.
run_cell() {
    local phase="$1"
    local method="$2"
    local dataset="$3"
    local log_file="$run_dir/${method}__${dataset}.log"
    local label="$method / $dataset"

    echo
    echo "[RUN ] [$phase] $label"
    echo "[RUN ] [$phase] $label  $(date --iso-8601=seconds 2>/dev/null || date)" >> "$sweep_log"

    local start_s
    start_s=$(date +%s)

    # Build the command as an array so quoting survives intact.
    local cmd
    cmd=(pixi run -e "$PIXI_ENV" dgadb run
         --method "$method"
         --datasets "$dataset"
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

    local rc=0
    "${cmd[@]}" > "$log_file" 2>&1 || rc=$?
    local elapsed=$(( $(date +%s) - start_s ))

    # Success detection. A successful run ends with the subprocess
    # logging "ExperimentRunner: Done." from ExperimentRunner.run(). A
    # subprocess crash leaves a Python traceback in the captured output
    # (stderr of the spawn process) and no "Done." line. See header
    # comment for why we cannot rely on "Experiment failed:" here.
    local result="FAIL"
    local reason=""
    if [[ $rc -ne 0 ]]; then
        if [[ "$TIMEOUT_SEC" != "0" && $rc -eq 124 ]]; then
            reason="timeout after ${TIMEOUT_SEC}s"
        else
            reason="pixi rc=$rc"
        fi
    elif grep -q "Traceback (most recent call last)" "$log_file"; then
        local first_err
        first_err=$(grep -A 20 "Traceback (most recent call last)" "$log_file" \
            | grep -m1 -E "^[A-Z][A-Za-z]*Error|^[A-Z][A-Za-z]*Exception" \
            | tr -d '\n' | cut -c1-200)
        if [[ -z "$first_err" ]]; then
            first_err="see traceback in log"
        fi
        reason="subprocess raised — $first_err"
    elif ! grep -q "ExperimentRunner: Done\." "$log_file"; then
        reason="no 'ExperimentRunner: Done.' marker — run did not reach completion"
    else
        result="PASS"
    fi

    # Append one row to the results TSV. Tabs are stripped from reason
    # so the TSV stays parseable even if grep picked up a message with
    # embedded tabs.
    local sanitized_reason
    sanitized_reason=$(printf '%s' "$reason" | tr '\t' ' ')
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$phase" "$method" "$dataset" "$result" "$elapsed" "$sanitized_reason" \
        >> "$results_tsv"

    if [[ "$result" == "PASS" ]]; then
        echo "[ OK ] [$phase] $label  (${elapsed}s)"
        echo "[ OK ] [$phase] $label  (${elapsed}s)" >> "$sweep_log"
    else
        echo "[FAIL] [$phase] $label  (${elapsed}s — $reason)"
        echo "[FAIL] [$phase] $label  (${elapsed}s — $reason — see $log_file)" >> "$sweep_log"
    fi
}

# Phase 1: methods axis. Every method × $PHASE1_DATASET.
if ! $SKIP_PHASE1; then
    echo
    echo "=== Phase 1: methods × $PHASE1_DATASET ==="
    echo "=== Phase 1: methods × $PHASE1_DATASET ===" >> "$sweep_log"
    for method in $METHODS; do
        run_cell "p1" "$method" "$PHASE1_DATASET"
    done
fi

# Phase 2: datasets axis. $PHASE2_METHOD × every dataset (skipping
# $PHASE1_DATASET to avoid redundancy with phase 1).
if ! $SKIP_PHASE2; then
    echo
    echo "=== Phase 2: $PHASE2_METHOD × datasets ==="
    echo "=== Phase 2: $PHASE2_METHOD × datasets ===" >> "$sweep_log"
    for dataset in $DATASETS; do
        # Skip the (phase2_method, phase1_dataset) cell only when phase 1
        # actually ran it — that is, when phase 1 was not skipped AND
        # $PHASE2_METHOD is one of the methods in $METHODS. Otherwise
        # phase 1 never touched this cell and we want to test it here.
        if [[ "$dataset" == "$PHASE1_DATASET" ]] \
           && ! $SKIP_PHASE1 \
           && contains_word "$PHASE2_METHOD" "$METHODS"; then
            echo "[SKIP] [p2] $PHASE2_METHOD / $dataset  (covered by phase 1)"
            echo "[SKIP] [p2] $PHASE2_METHOD / $dataset  (covered by phase 1)" >> "$sweep_log"
            continue
        fi
        run_cell "p2" "$PHASE2_METHOD" "$dataset"
    done
fi

# Summary table, rendered from the TSV.
echo
echo "=== smoke_test summary ==="
echo "=== smoke_test summary ===" >> "$sweep_log"
printf "%-5s %-12s %-16s %-6s %8s  %s\n" "phase" "method" "dataset" "result" "time" "reason"
printf "%-5s %-12s %-16s %-6s %8s  %s\n" "phase" "method" "dataset" "result" "time" "reason" >> "$sweep_log"
printf "%-5s %-12s %-16s %-6s %8s  %s\n" "-----" "------------" "----------------" "------" "--------" "------"
printf "%-5s %-12s %-16s %-6s %8s  %s\n" "-----" "------------" "----------------" "------" "--------" "------" >> "$sweep_log"

pass=0
fail=0
# Skip the header row (tail -n +2).
while IFS=$'\t' read -r phase method dataset result elapsed reason; do
    printf "%-5s %-12s %-16s %-6s %8s  %s\n" "$phase" "$method" "$dataset" "$result" "${elapsed}s" "$reason"
    printf "%-5s %-12s %-16s %-6s %8s  %s\n" "$phase" "$method" "$dataset" "$result" "${elapsed}s" "$reason" >> "$sweep_log"
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
    echo "=== smoke_test finished at $(date --iso-8601=seconds 2>/dev/null || date) ==="
} >> "$sweep_log"

if [[ $fail -gt 0 ]]; then
    exit 1
fi
exit 0
