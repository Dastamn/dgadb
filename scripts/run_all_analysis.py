#!/usr/bin/env python3
"""Master script to run all analysis experiments.

Orchestrates execution of:
- Experiment 4a: Spectral baseline analysis
- Experiment 4b: Spectral shift under anomaly injection
- Experiment 5: Multigraph multiplicity analysis
"""

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import typer

app = typer.Typer()

DATASETS = [
    "bitcoin-alpha",
    "bitcoin-otc",
    "as-topology",
    "digg-homo",
    "email-dnc",
    "uc-social",
]


def run_script(script_path: str, args: list[str], description: str) -> bool:
    """Run a Python script and return success status.

    Args:
        script_path: Path to the Python script.
        args: Command-line arguments to pass.
        description: Description for logging.

    Returns:
        True if script succeeded, False otherwise.
    """
    logger = logging.getLogger("run_all")
    logger.info(f"Starting: {description}")

    cmd = [sys.executable, script_path] + args
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description} (exit code {e.returncode})")
        return False


@app.command()
def main(
    n_workers: int = typer.Option(16, help="Number of parallel workers"),
    output_base: str = typer.Option("analysis-results", help="Base output directory"),
    experiment_dir: str = typer.Option(
        "experiment-results", help="Directory with benchmark results"
    ),
    skip_baseline: bool = typer.Option(False, help="Skip spectral baseline (4a)"),
    skip_shift: bool = typer.Option(False, help="Skip spectral shift (4b)"),
    skip_multigraph: bool = typer.Option(False, help="Skip multigraph analysis (5)"),
    datasets: list[str] = typer.Option(None, help="Specific datasets to analyze"),
) -> None:
    """Run all analysis experiments in sequence."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("run_all")

    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting all analysis experiments at {start_time.isoformat()}")

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    target_datasets = datasets if datasets else DATASETS
    dataset_args = []
    for ds in target_datasets:
        dataset_args.extend(["--datasets", ds])

    # Experiment 4a: Spectral Baseline
    if not skip_baseline:
        success = run_script(
            os.path.join(scripts_dir, "run_spectral_baseline.py"),
            [
                "--n-workers",
                str(n_workers),
                "--output-base",
                os.path.join(output_base, "spectral"),
            ]
            + dataset_args,
            "Experiment 4a: Spectral Baseline",
        )
        results["spectral_baseline"] = success

    # Experiment 4b: Spectral Shift
    if not skip_shift:
        success = run_script(
            os.path.join(scripts_dir, "run_spectral_shift.py"),
            [
                "--n-workers",
                str(n_workers),
                "--output-base",
                os.path.join(output_base, "spectral"),
            ]
            + dataset_args,
            "Experiment 4b: Spectral Shift",
        )
        results["spectral_shift"] = success

    # Experiment 5: Multigraph Analysis
    if not skip_multigraph:
        success = run_script(
            os.path.join(scripts_dir, "run_multigraph_analysis.py"),
            [
                "--n-workers",
                str(n_workers),
                "--output-base",
                os.path.join(output_base, "multigraph"),
                "--experiment-dir",
                experiment_dir,
            ]
            + dataset_args,
            "Experiment 5: Multigraph Analysis",
        )
        results["multigraph"] = success

    end_time = datetime.now(timezone.utc)
    duration = end_time - start_time

    logger.info("=" * 60)
    logger.info("Analysis Summary")
    logger.info("=" * 60)
    for experiment, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {experiment}: {status}")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Results saved to: {output_base}")

    # Exit with error if any experiment failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    app()
