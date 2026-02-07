#!/bin/bash
# Wrapper script to run scalability benchmark in the container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGADB_ROOT="${DGADB_ROOT:-$(dirname "$SCRIPT_DIR")}"
CONTAINER="${SCRIPT_DIR}/dgadb.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container image $CONTAINER not found."
    echo "Please build it first using: apptainer build $CONTAINER dgadb.def"
    exit 1
fi

# Set PYTHONPATH to include /app so that scripts.benchmark_scalability can be found
# and /app/src for the dgadb package.
# The container already has PYTHONPATH=/app/src:$PYTHONPATH.
export APPTAINERENV_PYTHONPATH="/app:/app/src:${PYTHONPATH}"
export APPTAINERENV_BASE_PATH="/app"

echo "Running scalability benchmark in container..."
apptainer exec --bind "${DGADB_ROOT}:/app" "$CONTAINER" python -m scripts.benchmark_scalability "$@"
