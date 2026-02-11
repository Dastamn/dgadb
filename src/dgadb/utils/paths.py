import os
from pathlib import Path


def get_project_root() -> str:
    """Return the project root directory.

    Resolution order:
    1. BASE_PATH environment variable (backwards-compatible).
    2. Walk up from this file to find the directory containing pyproject.toml.
       Works because the package is installed editable via pixi.
    """
    env = os.environ.get("BASE_PATH")
    if env:
        return env

    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return str(parent)

    raise RuntimeError(
        "Could not determine project root. Set the BASE_PATH environment variable "
        "or ensure pyproject.toml exists in an ancestor directory."
    )
