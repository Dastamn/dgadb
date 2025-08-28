#!/bin/bash
set -e

echo "Running all download scripts in scripts/ (except amazon.py and trace-theia.py)..."

for script in scripts/*.py; do
    base=$(basename "$script")
    if [[ "$base" == "amazon.py" || "$base" == "trace-theia.py" ]]; then
        echo "Skipping $base"
        continue
    fi

    echo "Running $script"
    python "$script"
done

echo "Success!"