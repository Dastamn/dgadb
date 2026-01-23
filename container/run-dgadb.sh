#!/bin/bash
# Wrapper script to run DGADB experiments in the container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGADB_ROOT="${DGADB_ROOT:-$(dirname "$SCRIPT_DIR")}"
CONTAINER="${SCRIPT_DIR}/dgadb.sif"

apptainer run --bind "${DGADB_ROOT}:/app" "$CONTAINER" "$@"
