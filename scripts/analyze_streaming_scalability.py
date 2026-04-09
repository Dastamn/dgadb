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

import json
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
    ("taddy", "enron"): "memory-bound",
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


@app.command()
def latency_cdf(
    sidecar_glob: Annotated[str, typer.Option(help="Glob pattern for sidecar JSONs")],
    output: Annotated[Path, typer.Option(help="Output PDF path")] = Path("tier_b_latency_cdf.pdf"),
) -> None:
    """Plot per-snapshot latency CDFs, one curve per (method, dataset) pair."""
    import glob

    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(sidecar_glob))
    if not paths:
        raise typer.BadParameter(f"No sidecars matched {sidecar_glob}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in paths:
        slug = Path(p).stem  # method__dataset
        data = json.loads(Path(p).read_text())
        latencies = sorted(data["per_snapshot_latency_ms"])
        if not latencies:
            continue
        n = len(latencies)
        ys = [(i + 1) / n for i in range(n)]
        ax.plot(latencies, ys, label=slug)

    ax.set_xscale("log")
    ax.set_xlabel("Per-snapshot latency (ms, log scale)")
    ax.set_ylabel("Cumulative fraction of snapshots")
    ax.axhline(0.95, linestyle="--", color="grey", linewidth=0.8)
    ax.axhline(0.99, linestyle=":", color="grey", linewidth=0.8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    typer.echo(f"Wrote {output}")


@app.command()
def cost_curve(
    sidecar_glob: Annotated[str, typer.Option(help="Glob pattern for sidecar JSONs")],
    output: Annotated[Path, typer.Option(help="Output PDF path")] = Path("tier_b_cost_curve.pdf"),
) -> None:
    """Plot per-snapshot latency vs. mean edge degree, one color per method.

    Linear-scaling methods (RustGraph, GraphSAGE) should appear flat; per-edge-
    subgraph methods (StrGNN) should be superlinear in mean degree. This is
    the figure that turns section 4.5's categorical compute-bound label into
    a quantitative cost mechanism.
    """
    import glob
    from collections import defaultdict

    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(sidecar_glob))
    if not paths:
        raise typer.BadParameter(f"No sidecars matched {sidecar_glob}")

    by_method: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in paths:
        slug = Path(p).stem  # method__dataset
        method = slug.split("__")[0]
        data = json.loads(Path(p).read_text())
        for lat, deg in zip(
            data["per_snapshot_latency_ms"], data["per_snapshot_mean_degree"]
        ):
            by_method[method].append((deg, lat))

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, points in sorted(by_method.items()):
        if not points:
            continue
        points.sort()
        xs = [d for d, _ in points]
        ys = [latency for _, latency in points]
        ax.scatter(xs, ys, s=8, alpha=0.5, label=method)

    ax.set_xlabel("Mean edge degree in snapshot")
    ax.set_ylabel("Per-snapshot latency (ms)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
