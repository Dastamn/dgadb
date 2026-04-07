"""Produce the appendix table and figures for the streaming-scalability rebuttal.

Reads the CSVs and sidecar JSONs written by ``scripts/benchmark_scalability.py``
and produces:
- ``tier_a_table.tex``: methods x datasets, cells show throughput (edges/s) or
  ``OOM (compute-bound)`` / ``OOM (memory-bound)`` (inherited from section 4.5).
- ``tier_b_latency_cdf.pdf``: per-snapshot latency CDFs, one panel per method.
- ``tier_b_cost_curve.pdf``: per-snapshot latency vs. mean edge degree.

Tasks 9 and 10 add the plot commands; this file currently exposes only
``tier-a-table``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

app = typer.Typer(pretty_exceptions_enable=False)


# OOM cells inherited verbatim from section 4.5 / Figure 3 of the published paper.
# Format: (method, dataset) -> "compute-bound" | "memory-bound".
# Extend as Tier A runs reveal additional ceilings.
KNOWN_OOM: dict[tuple[str, str], str] = {
    ("taddy", "as-topology"): "memory-bound",
    ("taddy", "epinions"): "memory-bound",
    ("generaldyg", "as-topology"): "memory-bound",
    ("generaldyg", "epinions"): "memory-bound",
    ("strgnn", "epinions"): "compute-bound",
    ("sad", "epinions"): "compute-bound",
    ("addgraph", "epinions"): "compute-bound",
}


@app.command()
def tier_a_table(
    csv_glob: Annotated[str, typer.Option(help="Glob pattern for benchmark CSVs")],
    output: Annotated[Path, typer.Option(help="Output .tex path")] = Path("tier_a_table.tex"),
) -> None:
    """Produce the Tier A LaTeX table from one or more benchmark CSVs."""
    import glob

    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise typer.BadParameter(f"No CSVs matched {csv_glob}")
    df = pl.concat([pl.read_csv(p) for p in paths])

    pivot = df.pivot(
        values="throughput_warmup_excluded",
        index="method",
        on="dataset",
        aggregate_function="median",
    )

    methods = sorted(pivot["method"].to_list())
    datasets = [c for c in pivot.columns if c != "method"]

    lines: list[str] = []
    lines.append("\\begin{tabular}{l" + "r" * len(datasets) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(datasets) + " \\\\")
    lines.append("\\midrule")
    for m in methods:
        row = pivot.filter(pl.col("method") == m).row(0, named=True)
        cells = []
        for d in datasets:
            tag = KNOWN_OOM.get((m, d))
            if tag is not None:
                cells.append(f"OOM ({tag})")
            else:
                v = row[d]
                cells.append(f"{v:,.0f}" if v is not None else "--")
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    output.write_text("\n".join(lines) + "\n")
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
