#!/bin/bash
#
# Run Spectral and Multigraph Analysis in Apptainer Container
# ============================================================
#
# This script runs the analysis pipeline inside the DGADB Apptainer container.
#
# Usage:
#   ./container/run-analysis.sh [OPTIONS]
#
# Options:
#   --n-workers N     Number of parallel workers (default: 8)
#   --skip-download   Skip data download check
#   --skip-baseline   Skip spectral baseline analysis (Exp 4a)
#   --skip-shift      Skip spectral shift analysis (Exp 4b)
#   --skip-multigraph Skip multigraph analysis (Exp 5)
#   --data-dir PATH   Custom data directory to bind mount
#   --cpus RANGE      CPU range for taskset (e.g., "0-7")
#
# Example:
#   ./container/run-analysis.sh --n-workers 16 --cpus 0-15
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER_IMAGE="$SCRIPT_DIR/dgadb.sif"

# Default settings
N_WORKERS=8
SKIP_DOWNLOAD=""
SKIP_BASELINE=""
SKIP_SHIFT=""
SKIP_MULTIGRAPH=""
DATA_DIR=""
CPUS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-workers)
            N_WORKERS="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --skip-baseline)
            SKIP_BASELINE="--skip-baseline"
            shift
            ;;
        --skip-shift)
            SKIP_SHIFT="--skip-shift"
            shift
            ;;
        --skip-multigraph)
            SKIP_MULTIGRAPH="--skip-multigraph"
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -27
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check container exists
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: Container image not found at $CONTAINER_IMAGE"
    echo ""
    echo "Build it first with:"
    echo "  cd $PROJECT_ROOT"
    echo "  apptainer build container/dgadb.sif container/dgadb.def"
    exit 1
fi

# Build bind mount options
BIND_OPTS="--bind $PROJECT_ROOT:/app"
if [ -n "$DATA_DIR" ]; then
    BIND_OPTS="$BIND_OPTS --bind $DATA_DIR:/app/data"
fi

# Build environment options
ENV_OPTS="--env OMP_NUM_THREADS=$N_WORKERS"
ENV_OPTS="$ENV_OPTS --env MKL_NUM_THREADS=$N_WORKERS"
ENV_OPTS="$ENV_OPTS --env OPENBLAS_NUM_THREADS=$N_WORKERS"

# Build the command
CMD="apptainer exec $BIND_OPTS $ENV_OPTS $CONTAINER_IMAGE"
CMD="$CMD /app/scripts/run_analysis_internal.sh"
CMD="$CMD --n-workers $N_WORKERS"
CMD="$CMD $SKIP_DOWNLOAD $SKIP_BASELINE $SKIP_SHIFT $SKIP_MULTIGRAPH"

# Add CPU pinning if specified
if [ -n "$CPUS" ]; then
    CMD="taskset -c $CPUS $CMD"
fi

echo "=============================================="
echo "DGADB Analysis Pipeline (Apptainer)"
echo "=============================================="
echo "Container: $CONTAINER_IMAGE"
echo "Project:   $PROJECT_ROOT"
echo "Workers:   $N_WORKERS"
[ -n "$CPUS" ] && echo "CPUs:      $CPUS"
[ -n "$DATA_DIR" ] && echo "Data dir:  $DATA_DIR"
echo "=============================================="
echo ""

# Run the command
exec $CMD
