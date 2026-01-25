#!/bin/bash
# Run Aim UI from the Apptainer container
#
# Usage:
#   ./aim-up.sh              # Uses default port 43800
#   ./aim-up.sh 8080         # Uses custom port

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PORT="${1:-43800}"

apptainer exec --bind "$REPO_ROOT:/app" "$SCRIPT_DIR/dgadb.sif" aim up --host 0.0.0.0 --port "$PORT"
