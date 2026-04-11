"""DyGADBench supplementary-materials browser.

Interactive marimo notebook for scanning the per-dataset CSV tables and
the sensitivity-analysis PDFs in this directory. Launch with:

    pixi run marimo edit supplementary-materials/browse.py

or

    pixi run marimo run supplementary-materials/browse.py
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    return Path, base64, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md(r"""
    # DyGADBench — Supplementary Materials Browser

    Interactive viewer for the per-dataset results (Table 3 in the
    paper) and the sensitivity-analysis plots (Figures 1b–1k). Use
    the widgets below to filter the results table, rank methods
    head-to-head at a fixed configuration, inspect a per-method
    heatmap, or browse the sensitivity PDFs.

    See `README.md` in this folder for the schema and dataset
    statistics.
    """)
    return


@app.cell
def _(Path):
    # Paths are resolved relative to the notebook file so the browser
    # works regardless of the shell's current working directory.
    SUPP_DIR = Path(__file__).resolve().parent
    TABLES_DIR = SUPP_DIR / "tables"
    PLOTS_DIR = SUPP_DIR / "plots"

    METHODS = [
        "sad", "taddy", "slade", "strgnn", "rustgraph",
        "gcn", "gat", "graphsage", "generaldyg", "addgraph",
    ]
    DATASETS = [
        "bitcoin-alpha", "bitcoin-otc", "email-dnc",
        "uc-social", "digg-homo", "as-topology",
    ]
    ANOM_TYPES = ["random", "bridge", "path", "burst", "clique"]
    ANOM_RATIOS = [0.01, 0.05, 0.1]
    ANOM_DURATIONS = [0.1, 0.5, 1.0]
    METRICS = ["rocauc", "auprc"]
    return (
        ANOM_DURATIONS,
        ANOM_RATIOS,
        ANOM_TYPES,
        DATASETS,
        METHODS,
        METRICS,
        PLOTS_DIR,
        TABLES_DIR,
    )


@app.cell
def _(METHODS, TABLES_DIR, mo, pl):
    # Load every per-method CSV into one combined dataframe. We cast
    # anom_duration and anom_ratio to Float64 because the CSV stores
    # 1.0 as "1" on some rows, which polars would otherwise infer as
    # Int64 and fail to merge with the 0.1/0.5 Float rows on concat.
    _dfs = []
    _missing = []
    for _m in METHODS:
        _path = TABLES_DIR / f"{_m}.csv"
        if _path.exists():
            _dfs.append(
                pl.read_csv(_path).with_columns(
                    pl.col("anom_duration").cast(pl.Float64),
                    pl.col("anom_ratio").cast(pl.Float64),
                )
            )
        else:
            _missing.append(_m)

    if not _dfs:
        raise FileNotFoundError(
            f"No method CSVs found under {TABLES_DIR}. "
            f"Expected one of: {METHODS}."
        )

    results = pl.concat(_dfs, how="vertical_relaxed")

    mo.md(
        f"Loaded **{results.height:,}** rows from "
        f"**{len(_dfs)}/{len(METHODS)}** method files"
        + (f" (missing: `{', '.join(_missing)}`)" if _missing else "")
        + "."
    )
    return (results,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Filterable results table
    """)
    return


@app.cell
def _(ANOM_DURATIONS, ANOM_RATIOS, ANOM_TYPES, DATASETS, METHODS, mo):
    method_select = mo.ui.multiselect(
        options=METHODS, value=METHODS, label="Methods"
    )
    dataset_select = mo.ui.multiselect(
        options=DATASETS, value=DATASETS, label="Datasets"
    )
    anom_type_select = mo.ui.multiselect(
        options=ANOM_TYPES, value=ANOM_TYPES, label="Anomaly types"
    )
    anom_ratio_select = mo.ui.multiselect(
        options=ANOM_RATIOS, value=ANOM_RATIOS, label="Injection rate R"
    )
    anom_duration_select = mo.ui.multiselect(
        options=ANOM_DURATIONS,
        value=ANOM_DURATIONS,
        label="Temporal span T",
    )

    mo.vstack(
        [
            mo.md(
                "Narrow the combined result table by any subset of the "
                "five dimensions. All widgets default to *everything*, "
                "so you see the full 2,700-row table until you start "
                "clicking things off."
            ),
            mo.hstack(
                [
                    method_select,
                    dataset_select,
                    anom_type_select,
                    anom_ratio_select,
                    anom_duration_select,
                ],
                justify="start",
                gap=1,
            ),
        ]
    )
    return (
        anom_duration_select,
        anom_ratio_select,
        anom_type_select,
        dataset_select,
        method_select,
    )


@app.cell
def _(
    anom_duration_select,
    anom_ratio_select,
    anom_type_select,
    dataset_select,
    method_select,
    mo,
    pl,
    results,
):
    filtered = (
        results.filter(pl.col("method").is_in(method_select.value))
        .filter(pl.col("dataset").is_in(dataset_select.value))
        .filter(pl.col("anom_type").is_in(anom_type_select.value))
        .filter(pl.col("anom_ratio").is_in(anom_ratio_select.value))
        .filter(pl.col("anom_duration").is_in(anom_duration_select.value))
    )

    if filtered.height == 0:
        _view = mo.md(
            "**No rows match the current filter.** "
            "Try widening the selection above."
        )
    else:
        _view = mo.vstack(
            [
                mo.md(
                    f"**{filtered.height:,}** rows match "
                    f"(out of {results.height:,} total)."
                ),
                mo.ui.table(
                    filtered,
                    selection=None,
                    show_column_summaries=False,
                    page_size=15,
                ),
            ]
        )
    _view
    return (filtered,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Aggregated summary
    """)
    return


@app.cell
def _(METRICS, mo):
    groupby_dim = mo.ui.dropdown(
        options=[
            "method",
            "dataset",
            "anom_type",
            "anom_ratio",
            "anom_duration",
        ],
        value="method",
        label="Group by",
    )
    summary_metric = mo.ui.radio(
        options=METRICS, value="rocauc", label="Metric"
    )
    summary_agg = mo.ui.radio(
        options=["mean", "median", "max", "min", "std"],
        value="mean",
        label="Aggregation",
    )

    mo.vstack(
        [
            mo.md(
                "Aggregate the **filtered** table from section 1 along "
                "one dimension. For example, group by `method` with "
                "`mean` `rocauc` to get the per-method average over "
                "whatever subset of datasets and anomaly configurations "
                "you selected above."
            ),
            mo.hstack(
                [groupby_dim, summary_metric, summary_agg],
                justify="start",
                gap=1,
            ),
        ]
    )
    return groupby_dim, summary_agg, summary_metric


@app.cell
def _(filtered, groupby_dim, mo, pl, summary_agg, summary_metric):
    _col = pl.col(summary_metric.value)
    _agg_expr = {
        "mean": _col.mean(),
        "median": _col.median(),
        "max": _col.max(),
        "min": _col.min(),
        "std": _col.std(),
    }[summary_agg.value]

    _out_col = f"{summary_metric.value}_{summary_agg.value}"

    if filtered.height == 0:
        # Keep summary_df defined so downstream cells don't see a
        # dangling reference when the filter is empty. We clear the
        # source dataframe and apply the same group_by so the schema
        # lines up with what a populated result would produce.
        summary_df = (
            filtered.clear()
            .group_by(groupby_dim.value)
            .agg(
                _agg_expr.alias(_out_col),
                pl.col(summary_metric.value).count().alias("n"),
            )
        )
        _view = mo.md(
            "*Nothing to aggregate — the filter above is empty.*"
        )
    else:
        summary_df = (
            filtered.group_by(groupby_dim.value)
            .agg(
                _agg_expr.alias(_out_col),
                pl.col(summary_metric.value).count().alias("n"),
            )
            .sort(_out_col, descending=True)
        )
        _view = mo.ui.table(
            summary_df,
            selection=None,
            show_column_summaries=False,
            page_size=15,
        )
    _view
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Head-to-head leaderboard
    """)
    return


@app.cell
def _(ANOM_DURATIONS, ANOM_RATIOS, ANOM_TYPES, DATASETS, METRICS, mo):
    h2h_dataset = mo.ui.dropdown(
        options=DATASETS, value=DATASETS[0], label="Dataset"
    )
    h2h_anom_type = mo.ui.dropdown(
        options=ANOM_TYPES, value=ANOM_TYPES[0], label="Anomaly type"
    )
    h2h_ratio = mo.ui.dropdown(
        options=ANOM_RATIOS,
        value=ANOM_RATIOS[1],
        label="Injection rate R",
    )
    h2h_duration = mo.ui.dropdown(
        options=ANOM_DURATIONS,
        value=ANOM_DURATIONS[1],
        label="Temporal span T",
    )
    h2h_metric = mo.ui.radio(
        options=METRICS, value="rocauc", label="Metric"
    )

    mo.vstack(
        [
            mo.md(
                "Fix every axis except **method** and see all ten "
                "methods ranked for this specific cell of the design."
            ),
            mo.hstack(
                [
                    h2h_dataset,
                    h2h_anom_type,
                    h2h_ratio,
                    h2h_duration,
                    h2h_metric,
                ],
                justify="start",
                gap=1,
            ),
        ]
    )
    return h2h_anom_type, h2h_dataset, h2h_duration, h2h_metric, h2h_ratio


@app.cell
def _(
    h2h_anom_type,
    h2h_dataset,
    h2h_duration,
    h2h_metric,
    h2h_ratio,
    mo,
    pl,
    results,
):
    leaderboard = (
        results.filter(
            (pl.col("dataset") == h2h_dataset.value)
            & (pl.col("anom_type") == h2h_anom_type.value)
            & (pl.col("anom_ratio") == h2h_ratio.value)
            & (pl.col("anom_duration") == h2h_duration.value)
        )
        .select(["method", h2h_metric.value])
        .sort(h2h_metric.value, descending=True)
    )

    _header = mo.md(
        f"**{h2h_dataset.value}** / `{h2h_anom_type.value}` / "
        f"R={h2h_ratio.value} / T={h2h_duration.value} — sorted by "
        f"**{h2h_metric.value}**"
    )

    if leaderboard.height == 0:
        _view = mo.vstack(
            [
                _header,
                mo.md("*No results for this configuration.*"),
            ]
        )
    else:
        _view = mo.vstack(
            [
                _header,
                mo.ui.table(
                    leaderboard,
                    selection=None,
                    show_column_summaries=False,
                    page_size=15,
                ),
            ]
        )
    _view
    return (leaderboard,)


@app.cell
def _(h2h_metric, leaderboard, plt):
    if leaderboard.height > 0:
        _fig, _ax = plt.subplots(figsize=(7, 3.5))
        _methods = leaderboard["method"].to_list()
        _values = leaderboard[h2h_metric.value].to_list()
        _bars = _ax.barh(_methods, _values, color="steelblue")
        _ax.invert_yaxis()
        _ax.set_xlabel(h2h_metric.value)
        _ax.set_xlim(0, 1)
        _ax.set_title(f"Ranked methods — {h2h_metric.value}")
        for _bar, _val in zip(_bars, _values):
            _ax.text(
                _val + 0.01,
                _bar.get_y() + _bar.get_height() / 2,
                f"{_val:.3f}",
                va="center",
                fontsize=8,
            )
        plt.tight_layout()
        _display = _fig
    else:
        _display = None
    _display
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Per-method heatmap
    """)
    return


@app.cell
def _(ANOM_TYPES, METHODS, METRICS, mo):
    hm_method = mo.ui.dropdown(
        options=METHODS, value="gcn", label="Method"
    )
    hm_anom_type = mo.ui.dropdown(
        options=ANOM_TYPES, value="random", label="Anomaly type"
    )
    hm_metric = mo.ui.radio(
        options=METRICS, value="rocauc", label="Metric"
    )

    mo.vstack(
        [
            mo.md(
                "Fix one method and one anomaly type, then visualise "
                "the whole `dataset × (R,T)` grid as a heatmap. Rows "
                "are the six datasets; columns are the nine "
                "(injection rate, temporal span) combinations."
            ),
            mo.hstack(
                [hm_method, hm_anom_type, hm_metric],
                justify="start",
                gap=1,
            ),
        ]
    )
    return hm_anom_type, hm_method, hm_metric


@app.cell
def _(
    ANOM_DURATIONS,
    ANOM_RATIOS,
    DATASETS,
    hm_anom_type,
    hm_method,
    hm_metric,
    mo,
    np,
    pl,
    plt,
    results,
):
    _slice = results.filter(
        (pl.col("method") == hm_method.value)
        & (pl.col("anom_type") == hm_anom_type.value)
    )

    _rt_cells = [(r, t) for r in ANOM_RATIOS for t in ANOM_DURATIONS]
    _mat = np.full((len(DATASETS), len(_rt_cells)), np.nan)

    for _di, _ds in enumerate(DATASETS):
        for _ci, (_r, _t) in enumerate(_rt_cells):
            _row = _slice.filter(
                (pl.col("dataset") == _ds)
                & (pl.col("anom_ratio") == _r)
                & (pl.col("anom_duration") == _t)
            )
            if _row.height > 0:
                _mat[_di, _ci] = _row[hm_metric.value][0]

    if np.isnan(_mat).all():
        _display = mo.md(
            "*No data for this (method, anomaly type) combination.*"
        )
    else:
        from matplotlib.colors import TwoSlopeNorm

        _fig, _ax = plt.subplots(figsize=(10, 4))
        # Diverging blue->white->red colormap centered at 0.5 — the
        # ROC-AUC random baseline. For AUPRC the center is less
        # meaningful, but the visual still makes "above/below random"
        # immediate at a glance for the dominant metric.
        _norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
        _im = _ax.imshow(_mat, cmap="RdBu_r", aspect="auto", norm=_norm)
        _ax.set_xticks(range(len(_rt_cells)))
        _ax.set_xticklabels(
            [f"R={r}\nT={t}" for r, t in _rt_cells], fontsize=8
        )
        _ax.set_yticks(range(len(DATASETS)))
        _ax.set_yticklabels(DATASETS)
        _ax.set_title(
            f"{hm_method.value} / {hm_anom_type.value} — "
            f"{hm_metric.value}"
        )
        for _di in range(len(DATASETS)):
            for _ci in range(len(_rt_cells)):
                _v = _mat[_di, _ci]
                if not np.isnan(_v):
                    # White text on the dark ends of the colormap,
                    # black text near the white midpoint.
                    _text_color = (
                        "black" if 0.30 <= _v <= 0.70 else "white"
                    )
                    _ax.text(
                        _ci,
                        _di,
                        f"{_v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=_text_color,
                    )
        _fig.colorbar(_im, ax=_ax, label=hm_metric.value)
        plt.tight_layout()
        _display = _fig
    _display
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Sensitivity-analysis plot browser
    """)
    return


@app.cell
def _(ANOM_TYPES, METRICS, mo):
    plot_metric = mo.ui.radio(
        options=METRICS, value="rocauc", label="Metric"
    )
    # Using a plain list here (rather than a {label: value} dict)
    # because mo.ui.radio's `value` argument refers to the dict KEY,
    # not the payload, so `value="1"` would KeyError. Easier to just
    # pass labels directly and parse the leading digit downstream.
    plot_experiment = mo.ui.radio(
        options=[
            "1 — varying injection rate R",
            "2 — varying temporal span T",
        ],
        value="1 — varying injection rate R",
        label="Experiment",
    )
    plot_anom_type = mo.ui.radio(
        options=ANOM_TYPES, value="random", label="Anomaly type"
    )

    mo.vstack(
        [
            mo.md(
                "Embedded viewer for the 20 sensitivity PDFs under "
                "`plots/`. Experiment 1 varies the injection rate R "
                "(aggregated over T and datasets); experiment 2 varies "
                "the temporal span T (aggregated over R and datasets). "
                "Pick a metric and anomaly type to swap the plot."
            ),
            mo.hstack(
                [plot_metric, plot_experiment, plot_anom_type],
                justify="start",
                gap=1,
            ),
        ]
    )
    return plot_anom_type, plot_experiment, plot_metric


@app.cell
def _(PLOTS_DIR, base64, mo, plot_anom_type, plot_experiment, plot_metric):
    # File naming convention used by the plot generator:
    #   exp_{1|2}_{a|b|c|d|e}_{type}.pdf
    # where the letter comes from the fixed ordering below.
    _type_to_letter = {
        "random": "a",
        "path": "b",
        "bridge": "c",
        "burst": "d",
        "clique": "e",
    }
    _letter = _type_to_letter[plot_anom_type.value]
    # plot_experiment.value is the full label "1 — varying ..." or
    # "2 — varying ..."; the leading character is the experiment ID.
    _exp_num = plot_experiment.value[0]
    _filename = f"exp_{_exp_num}_{_letter}_{plot_anom_type.value}.pdf"
    pdf_path = PLOTS_DIR / plot_metric.value / _filename

    if pdf_path.exists():
        _pdf_bytes = pdf_path.read_bytes()
        _pdf_b64 = base64.b64encode(_pdf_bytes).decode()
        _viewer = mo.Html(
            f'<embed src="data:application/pdf;base64,{_pdf_b64}" '
            f'type="application/pdf" width="100%" height="620px" />'
        )
        _download = mo.download(
            data=_pdf_bytes,
            filename=_filename,
            label=f"Download {_filename}",
        )
    else:
        _viewer = mo.md(f"**Plot not found:** `{pdf_path}`")
        _download = mo.md("")

    mo.vstack(
        [
            mo.md(
                f"**{pdf_path.relative_to(PLOTS_DIR.parent)}** — "
                f"{plot_metric.value}, experiment {_exp_num}, "
                f"`{plot_anom_type.value}`"
            ),
            _viewer,
            _download,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    *Data and schema documented in `supplementary-materials/README.md`.
    Tables were produced from Aim experiment runs; plots are
    generated by the paper's figure scripts.*
    """)
    return


if __name__ == "__main__":
    app.run()
