#!/bin/bash
#
# Spectral and Multigraph Analysis Runner
# =======================================
# This script downloads required datasets and runs the full analysis pipeline.
#
# Usage:
#   ./scripts/run_analysis_server.sh [OPTIONS]
#
# Options:
#   --n-workers N     Number of parallel workers (default: 8)
#   --skip-download   Skip data download check
#   --skip-baseline   Skip spectral baseline analysis (Exp 4a)
#   --skip-shift      Skip spectral shift analysis (Exp 4b)
#   --skip-multigraph Skip multigraph analysis (Exp 5)
#
# Requirements:
#   - pixi (for running Python scripts with dependencies)
#   - uv (for running data download scripts)
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default settings
N_WORKERS=${N_WORKERS:-8}
SKIP_DOWNLOAD=false
SKIP_BASELINE=false
SKIP_SHIFT=false
SKIP_MULTIGRAPH=false

# Datasets to analyze
DATASETS=(
    "bitcoin-alpha"
    "bitcoin-otc"
    "as-topology"
    "digg-homo"
    "email-dnc"
    "uc-social"
)

# Map dataset names to download scripts
declare -A DOWNLOAD_SCRIPTS
DOWNLOAD_SCRIPTS["bitcoin-alpha"]="bitcoin.py"
DOWNLOAD_SCRIPTS["bitcoin-otc"]="bitcoin.py"
DOWNLOAD_SCRIPTS["as-topology"]="as-topology.py"
DOWNLOAD_SCRIPTS["digg-homo"]="digg-homo.py"
DOWNLOAD_SCRIPTS["email-dnc"]="email-dnc.py"
DOWNLOAD_SCRIPTS["uc-social"]="uc-social.py"

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-workers)
            N_WORKERS="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip-shift)
            SKIP_SHIFT=true
            shift
            ;;
        --skip-multigraph)
            SKIP_MULTIGRAPH=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-workers N     Number of parallel workers (default: 8)"
            echo "  --skip-download   Skip data download check"
            echo "  --skip-baseline   Skip spectral baseline analysis (Exp 4a)"
            echo "  --skip-shift      Skip spectral shift analysis (Exp 4b)"
            echo "  --skip-multigraph Skip multigraph analysis (Exp 5)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed."
        exit 1
    fi
}

# =============================================================================
# Environment Setup
# =============================================================================

log_info "Setting up environment..."

cd "$PROJECT_ROOT"

# Check required tools
check_command "pixi"
check_command "uv"

# Set environment variables
export BASE_PATH="$PROJECT_ROOT"
export DATA_PATH="$PROJECT_ROOT"

log_info "Project root: $PROJECT_ROOT"
log_info "Workers: $N_WORKERS"

# =============================================================================
# Data Download
# =============================================================================

if [ "$SKIP_DOWNLOAD" = false ]; then
    log_info "Checking for required datasets..."

    # Track which download scripts need to run
    declare -A SCRIPTS_TO_RUN

    for dataset in "${DATASETS[@]}"; do
        data_file="$PROJECT_ROOT/data/$dataset/edges.parquet"

        if [ -f "$data_file" ]; then
            log_info "  ✓ $dataset - data exists"
        else
            log_info "  ✗ $dataset - data missing, will download"
            script="${DOWNLOAD_SCRIPTS[$dataset]}"
            SCRIPTS_TO_RUN["$script"]=1
        fi
    done

    # Run download scripts
    if [ ${#SCRIPTS_TO_RUN[@]} -gt 0 ]; then
        log_info "Downloading missing datasets..."

        for script in "${!SCRIPTS_TO_RUN[@]}"; do
            log_info "Running $script..."
            uv run python "$SCRIPT_DIR/$script"

            if [ $? -eq 0 ]; then
                log_success "Downloaded data via $script"
            else
                log_error "Failed to download data via $script"
                exit 1
            fi
        done
    else
        log_info "All datasets already exist."
    fi

    # Verify all data exists after download
    log_info "Verifying all datasets..."
    for dataset in "${DATASETS[@]}"; do
        data_file="$PROJECT_ROOT/data/$dataset/edges.parquet"

        if [ ! -f "$data_file" ]; then
            log_error "Dataset $dataset still missing after download!"
            exit 1
        fi
    done
    log_success "All datasets verified."
else
    log_info "Skipping data download check (--skip-download)"
fi

# =============================================================================
# Run Analysis
# =============================================================================

OUTPUT_DIR="$PROJECT_ROOT/analysis-results"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

log_info "Starting analysis pipeline..."
log_info "Output directory: $OUTPUT_DIR"

# Experiment 4a: Spectral Baseline
if [ "$SKIP_BASELINE" = false ]; then
    log_info "=========================================="
    log_info "Experiment 4a: Spectral Baseline Analysis"
    log_info "=========================================="

    pixi run python "$SCRIPT_DIR/run_spectral_baseline.py" \
        --n-workers "$N_WORKERS" \
        --output-base "$OUTPUT_DIR/spectral"

    if [ $? -eq 0 ]; then
        log_success "Spectral baseline analysis complete"
    else
        log_error "Spectral baseline analysis failed"
        exit 1
    fi
else
    log_info "Skipping spectral baseline (--skip-baseline)"
fi

# Experiment 4b: Spectral Shift
if [ "$SKIP_SHIFT" = false ]; then
    log_info "=========================================="
    log_info "Experiment 4b: Spectral Shift Analysis"
    log_info "=========================================="

    pixi run python "$SCRIPT_DIR/run_spectral_shift.py" \
        --n-workers "$N_WORKERS" \
        --output-base "$OUTPUT_DIR/spectral"

    if [ $? -eq 0 ]; then
        log_success "Spectral shift analysis complete"
    else
        log_error "Spectral shift analysis failed"
        exit 1
    fi
else
    log_info "Skipping spectral shift (--skip-shift)"
fi

# Experiment 5: Multigraph Analysis
if [ "$SKIP_MULTIGRAPH" = false ]; then
    log_info "=========================================="
    log_info "Experiment 5: Multigraph Analysis"
    log_info "=========================================="

    pixi run python "$SCRIPT_DIR/run_multigraph_analysis.py" \
        --n-workers "$N_WORKERS" \
        --output-base "$OUTPUT_DIR/multigraph" \
        --experiment-dir "$PROJECT_ROOT/experiment-results"

    if [ $? -eq 0 ]; then
        log_success "Multigraph analysis complete"
    else
        log_error "Multigraph analysis failed"
        exit 1
    fi
else
    log_info "Skipping multigraph analysis (--skip-multigraph)"
fi

# =============================================================================
# Summary
# =============================================================================

log_info "=========================================="
log_info "Analysis Pipeline Complete"
log_info "=========================================="
log_info ""
log_info "Results saved to: $OUTPUT_DIR"
log_info ""
log_info "Output structure:"
log_info "  analysis-results/"
log_info "  ├── spectral/"
log_info "  │   ├── {dataset}/baseline/     - Baseline spectral metrics"
log_info "  │   ├── {dataset}/anomalous/    - Spectral shift results"
log_info "  │   └── plots/                  - Cross-dataset visualizations"
log_info "  └── multigraph/"
log_info "      ├── {dataset}/              - Multiplicity statistics"
log_info "      └── correlation/            - DTDG/CTDG correlation analysis"
log_info ""
log_info "Plot formats: PNG (300 DPI) and SVG (vector)"

# List generated files
log_info ""
log_info "Generated plots:"
find "$OUTPUT_DIR" -name "*.png" -o -name "*.svg" 2>/dev/null | sort | while read -r file; do
    echo "  - ${file#$PROJECT_ROOT/}"
done

log_success "All done!"
