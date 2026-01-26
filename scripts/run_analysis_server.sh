#!/bin/bash
#
# Run Analysis with Pixi (non-container version)
# ==============================================
#
# This is a convenience wrapper for running the analysis without Apptainer.
# For server deployment with Apptainer, use: ./container/run-analysis.sh
#
# Usage:
#   ./scripts/run_analysis_server.sh [OPTIONS]
#
# Options:
#   --n-workers N     Number of parallel workers (default: 8)
#   --skip-download   Skip data download check
#   --skip-baseline   Skip spectral baseline analysis
#   --skip-shift      Skip spectral shift analysis
#   --skip-multigraph Skip multigraph analysis
#
# Requirements:
#   - pixi installed
#   - uv installed (for data download)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check for pixi
if ! command -v pixi &> /dev/null; then
    echo "ERROR: pixi is required but not installed."
    echo "Install it from: https://pixi.sh"
    exit 1
fi

export BASE_PATH="$PROJECT_ROOT"
export DATA_PATH="$PROJECT_ROOT"

# Run internal script with pixi
exec pixi run bash "$SCRIPT_DIR/run_analysis_internal.sh" "$@"
