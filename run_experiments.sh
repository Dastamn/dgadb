#!/usr/bin/env bash
# run_experiments.sh
# Usage examples:
#   METHODS="A B" DATASETS="D E" ./run_experiments.sh
#   METHODS="A B" DATASETS="D E" ./run_experiments.sh --num_samples 100
#   METHODS="A"   DATASETS="X Y" ./run_experiments.sh --num_samples 200 --num_gpus 2 --foo bar

set -u
set -o pipefail

# Require METHODS and DATASETS (space-separated)
if [[ -z "${METHODS:-}" || -z "${DATASETS:-}" ]]; then
  echo "Usage: METHODS=\"A B\" DATASETS=\"D E\" $0 [extra python flags like --num_samples 100 --num_gpus 2]"
  exit 1
fi

# Turn env vars into arrays
read -r -a METHODS_ARR <<< "$METHODS"
read -r -a DATASETS_ARR <<< "$DATASETS"

# Collect any extra flags given to this script to pass through to Python
EXTRA_ARGS=( "$@" )

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="run_logs/${TS}"
mkdir -p "$LOG_DIR"

for method in "${METHODS_ARR[@]}"; do
  for dataset in "${DATASETS_ARR[@]}"; do
    echo "Running method=$method dataset=$dataset ${EXTRA_ARGS:+extra args: ${EXTRA_ARGS[*]}}"
    log_file="${LOG_DIR}/method_${method}__dataset_${dataset}.log"

    if python -m src.dgadb.experiment.experiment \
         --method "$method" --dataset "$dataset" "${EXTRA_ARGS[@]}" |& tee "$log_file"; then
      echo "✓ Done: $method / $dataset"
    else
      echo "✗ Failed: $method / $dataset (see $log_file)"
    fi
    echo
  done
done

echo "Logs saved in: $LOG_DIR"
