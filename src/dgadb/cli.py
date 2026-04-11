"""Unified `dgadb` command-line entry point.

Thin wrapper that exposes the existing experiment runner, hyperparameter
tuner, and scalability profiler under a single command with subcommands:

    dgadb run ...           # runs experiments (dgadb.experiment.runner)
    dgadb tune ...          # runs Ray Tune sweeps (dgadb.experiment.tune)
    dgadb scalability ...   # runs the scalability benchmark

Each subcommand is the thinnest possible wrapper around a function or
script that already works. No new argument parsing, no new logic.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List

import typer

from dgadb.experiment.runner import run_experiment

app = typer.Typer(
    name="dgadb",
    help="DGADB — Dynamic Graph Anomaly Detection Benchmark CLI.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


# `dgadb run ...` — register the existing runner function directly so all
# options and defaults are inherited verbatim.
app.command(
    "run",
    help="Run one or more anomaly-detection experiments.",
)(run_experiment)


# `dgadb tune ...` — tune.py uses argparse at module scope, so we shell out
# with argv passthrough rather than re-declare its flags.
@app.command(
    "tune",
    help="Run Ray Tune hyperparameter sweeps (passthrough to dgadb.experiment.tune).",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def tune_cmd(ctx: typer.Context) -> None:
    cmd = [sys.executable, "-m", "dgadb.experiment.tune", *ctx.args]
    raise typer.Exit(code=subprocess.call(cmd))


# `dgadb scalability ...` — benchmark logic now lives in the package under
# src/dgadb/scalability/benchmark.py. Passthrough keeps it identical to the
# documented invocation.
@app.command(
    "scalability",
    help="Run the scalability benchmark (passthrough to dgadb.scalability.benchmark).",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def scalability_cmd(ctx: typer.Context) -> None:
    cmd = [sys.executable, "-m", "dgadb.scalability.benchmark", *ctx.args]
    raise typer.Exit(code=subprocess.call(cmd))


if __name__ == "__main__":
    app()
