# DGADB Apptainer Container

This directory contains files for building and running DGADB in an Apptainer container, enabling execution on Linux servers without local dependency management.

## Prerequisites

- [Apptainer](https://apptainer.org/) installed on the target Linux server
- Root or fakeroot privileges for building (not needed for running)

## Building the Container

From the repository root directory:

```bash
cd /path/to/dgadb
apptainer build container/dgadb.sif container/dgadb.def
```

This bakes all dependencies from `pixi.lock` into the image while keeping source code and data external via bind mounts.

## Usage

### Using the Wrapper Script

```bash
# Run an experiment
./container/run-dgadb.sh --method sad --dataset bitcoin-alpha

# With experiment name for Aim tracking
./container/run-dgadb.sh --method sad --dataset bitcoin-alpha --experiment-name my-exp
```

### Using Apptainer Directly

```bash
# Run an experiment
apptainer run --bind /path/to/dgadb:/app container/dgadb.sif --method sad --dataset bitcoin-alpha

# Interactive shell
apptainer shell --bind /path/to/dgadb:/app container/dgadb.sif

# View help
apptainer run-help container/dgadb.sif
```

### Custom Data Location

If your data is stored separately:

```bash
apptainer run \
    --bind /path/to/dgadb:/app \
    --bind /path/to/data:/app/data \
    container/dgadb.sif --method sad --dataset bitcoin-alpha
```

## CPU Resource Limiting

When running multiple experiments on a shared server, you can limit each run to specific CPUs using `taskset`. This provides hard isolation without requiring cgroups v2 configuration.

### Basic Usage

```bash
# Pin to CPUs 0-3
taskset -c 0-3 apptainer run --bind /path/to/dgadb:/app container/dgadb.sif --method sad --dataset bitcoin-alpha
```

### Combining with Thread Limits

For best results, also set thread environment variables to match:

```bash
taskset -c 0-3 apptainer run \
    --bind /path/to/dgadb:/app \
    --env OMP_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=4 \
    --env OPENBLAS_NUM_THREADS=4 \
    container/dgadb.sif --method sad --dataset bitcoin-alpha
```

### Running Multiple Isolated Experiments

Example script to run multiple experiments on different CPU sets:

```bash
#!/bin/bash
CPUS_PER_RUN=4
RUN_ID=$1
START_CPU=$((RUN_ID * CPUS_PER_RUN))
END_CPU=$((START_CPU + CPUS_PER_RUN - 1))

taskset -c "$START_CPU-$END_CPU" apptainer run \
    --bind /path/to/dgadb:/app \
    --env OMP_NUM_THREADS=$CPUS_PER_RUN \
    --env MKL_NUM_THREADS=$CPUS_PER_RUN \
    container/dgadb.sif "$@"
```

Usage:

```bash
./run_isolated.sh 0 --method sad --dataset bitcoin-alpha &   # CPUs 0-3
./run_isolated.sh 1 --method taddy --dataset bitcoin-alpha & # CPUs 4-7
./run_isolated.sh 2 --method slade --dataset bitcoin-alpha & # CPUs 8-11
```

## Aim UI (Experiment Tracking)

Run the Aim web UI from within the container to view experiment results.

### Using the Wrapper Script

```bash
./container/aim-up.sh        # Default port 43800
./container/aim-up.sh 8080   # Custom port
```

### Using Apptainer Directly

```bash
apptainer exec --bind /path/to/dgadb:/app container/dgadb.sif aim up --host 0.0.0.0 --port 43800
```

### Remote Access via SSH Port Forwarding

To access Aim running on a remote server from your local machine:

```bash
# On your local machine - create SSH tunnel
ssh -L 43800:localhost:43800 user@server

# On the server - start Aim
./container/aim-up.sh
```

Then open `http://localhost:43800` in your local browser.

### Running in Background

To keep Aim running after disconnecting:

```bash
# Using nohup
nohup ./container/aim-up.sh &

# Using tmux
tmux new -d -s aim './container/aim-up.sh'
```

## Verification

After building, verify the container works:

```bash
# Run built-in tests
apptainer test container/dgadb.sif

# Check CLI help
./container/run-dgadb.sh --help
```

## Design Notes

- **Source code**: Bind-mounted at `/app` for easy updates without rebuilding
- **Data**: Bind-mounted to avoid bloating the image
- **Dependencies**: Baked into the image via pixi for reproducibility
- **GPU support**: Currently CPU-only (matches current `pixi.lock`)

## Troubleshooting

### Permission errors

Ensure the bind-mounted directories are readable:

```bash
apptainer run --bind /path/to/dgadb:/app:ro container/dgadb.sif ...
```

### Missing datasets

Verify the dataset config exists and data is accessible at the bind-mounted location.
