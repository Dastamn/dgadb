#!/usr/bin/env bash

# python -m src.dgadb.experiment.experiment --method <M> --dataset <D>
# for all combinations of methods and datasets

set -u  #  continue on errors

# --- Edit these lists
METHODS=(A C D)
DATASETS=(B E F)
# ------------------------
# OR RUN:
# METHODS_OVERRIDE="A B" DATASETS_OVERRIDE="B C" ./run_experiments.sh


# overriding via environment variables, e.g.:
#   METHODS="A B" DATASETS="X Y" ./run_experiments.sh
if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  # e.g., METHODS_OVERRIDE="A B C"
  read -r -a METHODS <<< "$METHODS_OVERRIDE"
fi
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  # e.g., DATASETS_OVERRIDE="B D"
  read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="run_logs/run_logs_${TS}"
mkdir -p "$LOG_DIR"

failures=()
successes=0
total=0

echo "Starting experiment grid at $(date)"
echo "Methods:  ${METHODS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Logs will be written to: $LOG_DIR"
echo

for method in "${METHODS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    ((total++))
    log_file="${LOG_DIR}/method_${method}__dataset_${dataset}.log"
    cmd=( python -m src.dgadb.experiment.experiment --method "$method" --dataset "$dataset" )

    echo "[$total] Running: ${cmd[*]}"
    if "${cmd[@]}" &> "$log_file"; then
      echo " -> SUCCESS (log: $log_file)"
      ((successes++))
    else
      echo " -> FAILED  (log: $log_file)"
      failures+=("${method},${dataset}")
      # continue automatically to next combo
    fi
    echo
  done
done

echo "===== SUMMARY ====="
echo "Total runs:   $total"
echo "Succeeded:    $successes"
echo "Failed:       ${#failures[@]}"
if (( ${#failures[@]} > 0 )); then
  echo "Failures:"
  for item in "${failures[@]}"; do
    IFS=',' read -r m d <<< "$item"
    echo "  - method=$m, dataset=$d  (see: ${LOG_DIR}/method_${m}__dataset_${d}.log)"
  done
fi
echo "All logs: $LOG_DIR"
