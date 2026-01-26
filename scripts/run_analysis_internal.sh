#!/bin/bash
#
# Internal Analysis Runner (runs inside container or with pixi)
# =============================================================
#
# This script is called either from the Apptainer wrapper or directly with pixi.
# It handles data download and runs the analysis pipeline.
#
# Usage (inside container):
#   /app/scripts/run_analysis_internal.sh [OPTIONS]
#
# Usage (with pixi, from project root):
#   pixi run bash scripts/run_analysis_internal.sh [OPTIONS]
#
# Options:
#   --n-workers N     Number of parallel workers (default: 8)
#   --skip-download   Skip data download check
#   --skip-baseline   Skip spectral baseline analysis
#   --skip-shift      Skip spectral shift analysis
#   --skip-signature  Skip spectral signature analysis
#   --skip-multigraph Skip multigraph analysis
#

set -e

# =============================================================================
# Configuration
# =============================================================================

# Detect environment
if [ -n "$BASE_PATH" ]; then
    PROJECT_ROOT="$BASE_PATH"
elif [ -d "/app/src" ]; then
    PROJECT_ROOT="/app"  # Inside container
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

SCRIPT_DIR="$PROJECT_ROOT/scripts"

# Default settings
N_WORKERS=${N_WORKERS:-8}
SKIP_DOWNLOAD=false
SKIP_BASELINE=false
SKIP_SHIFT=false
SKIP_SIGNATURE=false
SKIP_MULTIGRAPH=false

# Datasets
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
        --skip-signature)
            SKIP_SIGNATURE=true
            shift
            ;;
        --skip-multigraph)
            SKIP_MULTIGRAPH=true
            shift
            ;;
        -h|--help)
            head -25 "$0" | tail -22
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

# =============================================================================
# Environment Setup
# =============================================================================

log_info "Setting up environment..."

cd "$PROJECT_ROOT"

export BASE_PATH="$PROJECT_ROOT"
export DATA_PATH="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

log_info "Project root: $PROJECT_ROOT"
log_info "Workers: $N_WORKERS"

# Detect Python command (inside container vs pixi)
if [ -f "/build/.pixi/envs/default/bin/python" ]; then
    PYTHON="/build/.pixi/envs/default/bin/python"
    log_info "Using container Python: $PYTHON"
elif command -v python &> /dev/null; then
    PYTHON="python"
    log_info "Using system Python"
else
    log_error "Python not found!"
    exit 1
fi

# =============================================================================
# Data Download
# =============================================================================

if [ "$SKIP_DOWNLOAD" = false ]; then
    log_info "Checking for required datasets..."

    declare -A SCRIPTS_TO_RUN

    for dataset in "${DATASETS[@]}"; do
        data_file="$PROJECT_ROOT/data/$dataset/edges.parquet"

        if [ -f "$data_file" ]; then
            log_info "  ✓ $dataset"
        else
            log_info "  ✗ $dataset - missing"
            script="${DOWNLOAD_SCRIPTS[$dataset]}"
            SCRIPTS_TO_RUN["$script"]=1
        fi
    done

    if [ ${#SCRIPTS_TO_RUN[@]} -gt 0 ]; then
        log_info "Downloading missing datasets..."

        for script in "${!SCRIPTS_TO_RUN[@]}"; do
            log_info "Running $script..."
            $PYTHON "$SCRIPT_DIR/$script"

            if [ $? -eq 0 ]; then
                log_success "Downloaded via $script"
            else
                log_error "Failed: $script"
                exit 1
            fi
        done
    else
        log_info "All datasets present."
    fi

    # Verify
    for dataset in "${DATASETS[@]}"; do
        data_file="$PROJECT_ROOT/data/$dataset/edges.parquet"
        if [ ! -f "$data_file" ]; then
            log_error "Dataset $dataset still missing!"
            exit 1
        fi
    done
    log_success "All datasets verified."
else
    log_info "Skipping data check (--skip-download)"
fi

# =============================================================================
# Run Analysis
# =============================================================================

OUTPUT_DIR="$PROJECT_ROOT/analysis-results"

log_info "Starting analysis..."
log_info "Output: $OUTPUT_DIR"

# Experiment 4a: Spectral Baseline
if [ "$SKIP_BASELINE" = false ]; then
    log_info "=========================================="
    log_info "Experiment 4a: Spectral Baseline"
    log_info "=========================================="

    $PYTHON "$SCRIPT_DIR/run_spectral_baseline.py" \
        --n-workers "$N_WORKERS" \
        --output-base "$OUTPUT_DIR/spectral"

    [ $? -eq 0 ] && log_success "Spectral baseline complete" || { log_error "Failed"; exit 1; }
else
    log_info "Skipping spectral baseline"
fi

# Experiment 4b: Spectral Shift
if [ "$SKIP_SHIFT" = false ]; then
    log_info "=========================================="
    log_info "Experiment 4b: Spectral Shift"
    log_info "=========================================="

    $PYTHON "$SCRIPT_DIR/run_spectral_shift.py" \
        --n-workers "$N_WORKERS" \
        --output-base "$OUTPUT_DIR/spectral"

    [ $? -eq 0 ] && log_success "Spectral shift complete" || { log_error "Failed"; exit 1; }
else
    log_info "Skipping spectral shift"
fi

# Experiment 4c: Spectral Signature
if [ "$SKIP_SIGNATURE" = false ]; then
    log_info "=========================================="
    log_info "Experiment 4c: Spectral Signature"
    log_info "=========================================="

    $PYTHON "$SCRIPT_DIR/run_spectral_signature.py" \
        --n-workers "$N_WORKERS" \
        --output-base "$OUTPUT_DIR/spectral"

    [ $? -eq 0 ] && log_success "Spectral signature complete" || { log_error "Failed"; exit 1; }
else
    log_info "Skipping spectral signature"
fi

# Experiment 5: Multigraph Analysis
if [ "$SKIP_MULTIGRAPH" = false ]; then
    log_info "=========================================="
    log_info "Experiment 5: Multigraph Analysis"
    log_info "=========================================="

    $PYTHON "$SCRIPT_DIR/run_multigraph_analysis.py" \
        --n-workers "$N_WORKERS" \
        --output-base "$OUTPUT_DIR/multigraph" \
        --experiment-dir "$PROJECT_ROOT/experiment-results"

    [ $? -eq 0 ] && log_success "Multigraph analysis complete" || { log_error "Failed"; exit 1; }
else
    log_info "Skipping multigraph analysis"
fi

# =============================================================================
# Summary
# =============================================================================

log_info "=========================================="
log_info "Analysis Complete"
log_info "=========================================="
log_info "Results: $OUTPUT_DIR"
log_info ""
log_info "Generated files:"
find "$OUTPUT_DIR" \( -name "*.png" -o -name "*.svg" -o -name "*.json" \) 2>/dev/null | wc -l | xargs -I {} echo "  {} files"

log_success "Done!"
