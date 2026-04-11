#!/usr/bin/env python3
"""Render dgadb's API documentation with pdoc.

Why this is a script and not a one-liner pixi task:

1. Some vendored modules under ``dgadb.models`` (notably the StrGNN
   subtree, which embeds an upstream ``pytorch_DGCNN`` snapshot) call
   ``argparse.parse_known_args()`` at *import time*. Running pdoc as a
   shell command means pdoc's own ``-o`` / ``--output-directory`` flag
   ends up in ``sys.argv`` while those modules are imported, and the
   short ``-o`` collides with DGCNN's ``-out_dim`` argument, raising
   ``argparse.ArgumentError`` and aborting the build. We work around
   this by clearing ``sys.argv`` before any dgadb module is imported
   and by calling the pdoc API directly instead of the CLI.

2. pdoc walks every submodule of any package it is given, so passing
   a parent like ``dgadb.models`` would crash on legacy directories
   (``dgadb.models.GeneralDYG.option`` and friends) that fail to
   import for unrelated reasons (missing optional dependencies like
   ``tensorboard``). Listing the importable modules explicitly is the
   most reliable way to keep the build green without rewriting
   vendored code or spreading ``__all__`` filters across every legacy
   ``__init__.py``.

Run via: ``pixi run -e dev docs`` (writes to ``docs/api/``).
"""

from __future__ import annotations

import sys

# Step 1: clear argv before importing pdoc or anything that imports a
# dgadb model. See note (1) in the module docstring.
_orig_argv = sys.argv
sys.argv = [_orig_argv[0]] if _orig_argv else [""]

import pdoc  # noqa: E402  (intentional ordering — see above)
import pdoc.render  # noqa: E402
from pathlib import Path  # noqa: E402


# Curated list of importable, current modules. Excludes legacy model
# directories (GAT/, GeneralDYG/, RustGraph/, SAD/, SLADE/, TADDY/,
# baseline/{netwalk,Node2Vec,TGAT,...}) and script-style files inside
# *_new directories (e.g. rustgraph_new/main.py imports tensorboard).
MODULES = [
    # CLI entry point
    "dgadb.cli",
    # Experiment orchestration + callbacks (whole subpackage walks cleanly)
    "dgadb.experiment",
    # Storage core (whole subpackage walks cleanly)
    "dgadb.storage",
    # Preprocessing — only the live anomaly injection module + the pipeline
    # subpackage. Older sibling files (anomaly_injector.py, snapshotting.py,
    # splitting.py, structural.py, temporal.py, anomaly_generation.py,
    # add_graph_temporary.py) are legacy and excluded.
    "dgadb.preprocessing.anomaly_injection",
    "dgadb.preprocessing.pipeline",
    "dgadb.preprocessing.normalization",
    # Streaming-scalability profiler + benchmark
    "dgadb.scalability",
    # Evaluation
    "dgadb.evaluation",
    # Data builder helpers
    "dgadb.data.builder",
    "dgadb.data.dataset",
    # Common utilities
    "dgadb.utils",
    # Model interface and the current model adapters. Each is listed by
    # leaf module rather than by parent package, because pdoc would
    # otherwise walk into broken legacy siblings (GeneralDYG/option.py,
    # rustgraph_new/main.py, StrGNN/detection/Main.py, ...).
    "dgadb.models.base",
    "dgadb.models.common",
    "dgadb.models.utils",
    "dgadb.models.baseline.gnn",
    "dgadb.models.addgraph.addgraph",
    "dgadb.models.sad_new.sad",
    "dgadb.models.slade_new.slade",
    "dgadb.models.taddy_new.taddy",
    "dgadb.models.generaldyg_new.generaldyg",
    "dgadb.models.rustgraph_new.rustgraph",
    "dgadb.models.StrGNN.strgnn",
]


def main(out_dir: str = "docs/api") -> None:
    """Render the curated module list to ``out_dir`` as HTML."""
    pdoc.render.configure(
        docformat="google",
        mermaid=True,
        show_source=True,
        footer_text="dgadb — Dynamic Graph Anomaly Detection Benchmark",
    )

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdoc.pdoc(*MODULES, output_directory=output_path)


if __name__ == "__main__":
    # Optional CLI: pass an output directory as the first argument.
    # We've already cleared the original argv above, so reach back to
    # ``_orig_argv`` to honour user overrides.
    target = _orig_argv[1] if len(_orig_argv) > 1 else "docs/api"
    main(target)
