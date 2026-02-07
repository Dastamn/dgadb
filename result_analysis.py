import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import polars.selectors as cs
    return pl, plt, sns


@app.cell
def _(pl):
    complete_version_path  = "MASTER_roc_auc_subsettest_final_final_long_form.csv"
    complete_df = pl.read_csv(complete_version_path)
    return (complete_df,)


@app.cell
def _(complete_df):
    complete_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Big table
    The following cell generates the big table that appears in the experiments section.
    """)
    return


@app.cell
def _(complete_df, pl):
    def merge_mean_and_std_cols(df: pl.DataFrame) -> pl.DataFrame:
        anomaly_types = ["random", "path", "bridge", "burst", "clique"]
        for anom_type in anomaly_types:
            df = df.with_columns(
                pl.col(f"mean_roc_auc_{anom_type}").round(2),
                pl.col(f"std_roc_auc_{anom_type}").round(2),
            )
            df = df.with_columns(
                pl.format(
                    "${} \pm {}$",
                    pl.col(f"mean_roc_auc_{anom_type}"),
                    pl.col(f"std_roc_auc_{anom_type}"),
                ).alias(anom_type)
            )
        relevant_columns = ["method"] + anomaly_types
        return df.select(relevant_columns)


    big_table = complete_df.group_by(["method", "anomaly_type"]).agg(
        pl.mean("roc_auc").alias("mean_roc_auc"),
        pl.std("roc_auc").alias("std_roc_auc"),
        pl.quantile("roc_auc", 0.9).alias("p90_roc_auc"),
    )
    # big_table = big_table.pivot(
    #     "anomaly_type", index="method", values=["mean_roc_auc", "std_roc_auc", "p90_roc_auc"]
    # )

    # need to unpivot the mean, std, and p90.
    big_table = big_table.rename(
        {"mean_roc_auc": "mean", "std_roc_auc": "std", "p90_roc_auc": "p90"}
    )
    big_table = big_table.unpivot(
        ["mean", "std", "p90"],
        index=["method", "anomaly_type"],
        variable_name="quantile",
        value_name="roc_auc",
    )
    # big_table = big_table.with_columns(cs.numeric().round(2))
    big_table = big_table.to_pandas().pivot_table(
        index="method",
        columns=["anomaly_type", "quantile"],
        values="roc_auc",
        aggfunc="first",
    )
    # big_table = merge_mean_and_std_cols(big_table)
    print(big_table.to_latex(float_format="{:.2f}".format, escape=True))
    # print(big_table.to_pandas().to_latex(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 1

    Let's try to define a common colour palette for the whole paper.
    """)
    return


@app.cell
def _(sns):
    method_names = ["strgnn", "taddy", "sad", "addgraph", "rustgraph"]
    palette = sns.color_palette("colorblind", n_colors=len(method_names))
    method_colors = {name: color for name, color in zip(method_names, palette)}

    anomaly_types = ["random", "path", "bridge", "burst", "clique"]
    return anomaly_types, method_names, palette


@app.cell
def _(anomaly_types, complete_df, method_names, palette, pl, plt, sns):
    exp_1_data = complete_df.filter(pl.col("method").is_in(method_names))


    def plot_exp_1():
        fig = plt.figure(figsize=(18, 4))

        subfigs_top = fig.subfigures(1, 3, wspace=0.15)
        axes_top = [subfig.subplots(1, 1) for subfig in subfigs_top]

        subfigs_bottom = fig.subfigures(1, 3, wspace=0.15)

        # Create figure-level grid specification for centering
        gs = fig.add_gridspec(2, 3)

        # Clear and use a simpler approach with gridspec
        plt.close("all")  # Close previous figure

        fig = plt.figure(figsize=(18, 4))
        gs = fig.add_gridspec(2, 3)

        # Row 1: 3 plots (columns 0, 1, 2)
        axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(3)]

        # Row 2: 2 plots centered (columns 0 and 1)
        axes_row2 = [fig.add_subplot(gs[1, i]) for i in range(2)]

        axes = axes_row1 + axes_row2

        for idx, ax in enumerate(axes):
            fig_single = plt.figure(figsize=(5, 3))
            ax_single = fig_single.add_subplot(111)

            # Plot the same data
            an_type = anomaly_types[idx]
            df_filtered = exp_1_data.filter(pl.col("anomaly_type") == an_type)

            sns.lineplot(
                data=df_filtered,
                x="anomaly_rate",
                y="roc_auc",
                hue="method",  # You can use: dataset, duration, anomaly_rate, etc.
                palette=palette,  # For the hue variable, can be any seaborn palette
                ax=ax_single,
                errorbar="ci",
                marker="o",
                markersize=10,
            )

            if idx == 0:  # Keep legend only on first plot
                wanted = ["strgnn", "sad", "rustgraph", "taddy", "addgraph"]  
                handles, labels = ax_single.get_legend_handles_labels()
                order_map = {label: idx for idx, label in enumerate(labels)}
                ax_single.legend([handles[order_map[l]] for l in wanted], wanted, title="Method", fontsize=14, title_fontsize=16, ncol=2)
            else:
                legend = ax_single.get_legend()
                if legend:
                    legend.remove()

            # Apply the same global y-limits
            ax_single.set_ylim(0, 1)
            ax_single.set_xlabel("$\mathcal{R}$", fontsize=20)
            ax_single.set_ylabel("ROC AUC", fontsize=20)
            ax_single.tick_params(axis="both", labelsize=18)
            # ax_single.set_title(
            #     f"{an_type}".capitalize(), fontsize=14, fontweight="bold"
            # )

            # NO label text added here either - LaTeX subcaption will handle it
            # plt.tight_layout()
            fig_single.savefig(
                f"paper_figures/exp_1_{chr(97 + idx)}_{an_type}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_single)


    plot_exp_1()
    return


@app.cell
def _(complete_df, pl, sns):
    new_exp_1_data = (complete_df
                  .filter(pl.col("method").is_in(["strgnn", "sad", "taddy", "rustgraph", "addgraph"]))
                 )

    new_exp_1_plot = sns.relplot(
        data=new_exp_1_data,
        kind="line",
        x="anomaly_rate",
        y="roc_auc",
        hue="method",
        # row="duration",
        col="anomaly_type",
        marker="o",
        markersize=10
    )


    def set_x_axis(x_label, g):
        g.ax

    new_exp_1_plot.savefig("paper_figures/new_experiment_1_frequency", dpi=900)

    new_exp_1_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 2
    """)
    return


@app.cell
def _(anomaly_types, complete_df, method_names, palette, pl, plt, sns):
    exp_2_data = complete_df.filter(pl.col("method").is_in(method_names),
                                   pl.col("anomaly_rate") == 0.05,
                                   pl.col("duration").is_in([0.1, 1.0]))

    label_offset = 5

    def plot_exp_2():
        fig = plt.figure(figsize=(18, 4))

        subfigs_top = fig.subfigures(1, 3, wspace=0.15)
        axes_top = [subfig.subplots(1, 1) for subfig in subfigs_top]

        subfigs_bottom = fig.subfigures(1, 3, wspace=0.15)

        # Create figure-level grid specification for centering
        gs = fig.add_gridspec(2, 3)

        # Clear and use a simpler approach with gridspec
        plt.close("all")  # Close previous figure

        fig = plt.figure(figsize=(18, 4))
        gs = fig.add_gridspec(2, 3)

        # Row 1: 3 plots (columns 0, 1, 2)
        axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(3)]

        # Row 2: 2 plots centered (columns 0 and 1)
        axes_row2 = [fig.add_subplot(gs[1, i]) for i in range(2)]

        axes = axes_row1 + axes_row2

        for idx, ax in enumerate(axes):
            fig_single = plt.figure(figsize=(5, 3))
            ax_single = fig_single.add_subplot(111)

            # Plot the same data
            an_type = anomaly_types[idx]
            df_filtered = exp_2_data.filter(pl.col("anomaly_type") == an_type)

            sns.lineplot(
                data=df_filtered,
                x="duration",
                y="roc_auc",
                hue="method",  # You can use: dataset, duration, anomaly_rate, etc.
                palette=palette,  # For the hue variable, can be any seaborn palette
                ax=ax_single,
                errorbar="ci",
                marker="o",
                markersize=10
            )

            legend = ax_single.get_legend()
            if legend:
                legend.remove()

            # Apply the same global y-limits
            ax_single.set_ylim(0, 1)
            ax_single.set_xlabel("$\mathcal{T}$", fontsize=20)
            ax_single.set_ylabel("ROC AUC", fontsize=20)
            ax_single.tick_params(axis='both', labelsize=18)
            # ax_single.set_title(
            #     f"{an_type}".capitalize(), fontsize=14, fontweight="bold"
            # )

            # NO label text added here either - LaTeX subcaption will handle it
            # plt.tight_layout()
            fig_single.savefig(
                f"paper_figures/exp_2_{chr(97 + idx)}_{an_type}.pdf", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_single)

    plot_exp_2()
    return


@app.cell
def _(complete_df, pl, sns):
    new_exp_2_data = (complete_df
        .filter(pl.col("anomaly_rate") == 0.05)
        .filter(pl.col("method").is_in(["strgnn", "taddy", "sad", "addgraph", "rustgraph"]))
        .filter(pl.col("duration").is_in([0.1, 1.0]))
                  # .group_by(["method", "anom_type", "anom_ratio", "anom_duration"]).agg(pl.mean("roc_auc"))
    )

    new_exp_2_plot = sns.relplot(
        data=new_exp_2_data,
        kind=f"line",
        x="duration",
        y="roc_auc",
        hue="method",
        col="anomaly_type"
    )



    new_exp_2_plot.savefig("paper_figures/new_experiment_2_temporal_span", dpi=900)

    new_exp_2_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 3
    """)
    return


@app.cell
def _(complete_df, method_names, pl, plt, sns):
    exp_3_data = complete_df.filter(pl.col("method").is_in(method_names),
                                   pl.col("anomaly_rate") == 0.05,)
    ordered_types = ["clique", "burst", "path", "bridge", "random"]

    def plot_exp_3():
        fig = plt.figure(figsize=(10, 3))

        ax = sns.boxplot(
            data=exp_3_data,
            x="anomaly_type",
            y="roc_auc",
            order=ordered_types,
            color="#6c7a89",
            boxprops=dict(alpha=.45)
        )


        g = sns.stripplot(
            data=exp_3_data,
            x="anomaly_type",
            y="roc_auc",
            order=ordered_types,
            ax=ax,
            color="k",
            size=3,
            jitter=True,
            dodge=True
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC AUC", fontsize=14)
        ax.set_xlabel("", fontsize=14)
        ax.tick_params(axis='both', labelsize=20)

        plt.tight_layout()
        plt.savefig("paper_figures/exp_3_overall.pdf")

    plot_exp_3()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 4
    """)
    return


@app.cell
def _(complete_df, method_names, palette, pl, plt, sns):
    exp4_data = (
        complete_df.filter(pl.col("anomaly_rate") == 0.05)
        .filter(pl.col("duration") == 0.5)
        .filter(pl.col("anomaly_type").is_in(["path", "random", "bridge"]))
        .filter(pl.col("method").is_in(method_names))
    )

    exp4_ordered_types = ["random", "path", "bridge"]
    exp4_datasets = (
        exp4_data.select("dataset").unique().sort("dataset").to_series().to_list()
    )


    def plot_exp_4():
        for idx, dataset in enumerate(exp4_datasets):
            fig_single = plt.figure(figsize=(5, 3))
            ax = fig_single.add_subplot(111)

            df_filtered = exp4_data.filter(pl.col("dataset") == dataset)

            sns.barplot(
                data=df_filtered,
                x="anomaly_type",
                y="roc_auc",
                hue="method",
                order=exp4_ordered_types,
                palette=palette,
                ax=ax,
                errorbar="ci",
            )

            if idx == 0:  # Keep legend only on first plot
                wanted = ["strgnn", "sad", "rustgraph", "taddy", "addgraph"]
                handles, labels = ax.get_legend_handles_labels()
                order_map = {label: idx for idx, label in enumerate(labels)}
                ax.legend(
                    [handles[order_map[l]] for l in wanted],
                    wanted,
                    title="Method",
                    fontsize=14,
                    loc="lower center",
                    title_fontsize=16,
                    ncol=2,
                )
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
            # if idx == 0:
            #     ax.legend(title="Method", fontsize=10, title_fontsize=12)
            # else:
            #     legend = ax.get_legend()
            #     if legend:
            #         legend.remove()

            ax.set_ylim(0, 1)
            ax.set_xlabel("", fontsize=14)
            ax.set_ylabel("ROC AUC", fontsize=20)
            ax.set_xticklabels([t.capitalize() for t in exp4_ordered_types])
            ax.tick_params(axis="both", labelsize=20)

            fig_single.savefig(
                f"paper_figures/exp_4_{chr(97 + idx)}_{dataset}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_single)


    plot_exp_4()
    return


@app.cell
def _(complete_df, method_names, palette, pl, plt, sns):
    exp5_data = (complete_df
        .filter(pl.col("anomaly_rate") == 0.05)
        .filter(pl.col("duration") == 1.0)
        .filter(pl.col("anomaly_type").is_in(["burst", "clique"]))
        .filter(pl.col("method").is_in(method_names))
    )

    exp5_ordered_types = ["burst", "clique"]
    exp5_datasets = exp5_data.select("dataset").unique().sort("dataset").to_series().to_list()

    def plot_exp_5():
        for idx, dataset in enumerate(exp5_datasets):
            fig_single = plt.figure(figsize=(5, 3))
            ax = fig_single.add_subplot(111)

            df_filtered = exp5_data.filter(pl.col("dataset") == dataset)

            sns.barplot(
                data=df_filtered,
                x="anomaly_type",
                y="roc_auc",
                hue="method",
                order=exp5_ordered_types,
                palette=palette,
                ax=ax,
                errorbar="ci",
            )

            # if idx == 0:
            #     ax.legend(title="Method", fontsize=10, title_fontsize=12)
            # else:
            legend = ax.get_legend()
            if legend:
                legend.remove()

            ax.set_ylim(0, 1)
            ax.set_xlabel("", fontsize=14)
            ax.set_ylabel("ROC AUC", fontsize=20)
            ax.set_xticklabels([t.capitalize() for t in exp5_ordered_types])
            ax.tick_params(axis="both", labelsize=20)

            fig_single.savefig(
                f"paper_figures/exp_5_{chr(97 + idx)}_{dataset}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_single)

    plot_exp_5()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scalability Analysis (RQ3)

    Computes every number in `reports/scalability/scalability_section.tex` from the raw data:

    - `MASTER_roc_auc_subsettest_final_final_long_form.csv` — test-set ROC-AUC (270 conditions per method)
    - `benchmark_results/all_methods_except_strgnn.json` — profiling (all methods except StrGNN)
    - `benchmark_results/strgnn_results.json` — profiling (StrGNN)
    """)
    return


@app.cell
def _(pl):
    import json

    with open("benchmark_results/all_methods_except_strgnn.json") as f:
        _bench_main = json.load(f)
    with open("benchmark_results/strgnn_results.json") as f:
        _bench_strgnn = json.load(f)

    bench_df = pl.DataFrame(_bench_main + _bench_strgnn)
    return (bench_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Table: Efficiency vs. Detection Quality

    | Column | Source |
    |--------|--------|
    | Edges/s | mean `edges_per_sec` across 6 datasets (benchmark JSON) |
    | GPU (MB) | mean `peak_gpu_mb` across 6 datasets (benchmark JSON) |
    | Inference (s) | `inference_time_sec` on as-topology only (benchmark JSON) |
    | AUC | mean `roc_auc` across all 270 conditions (CSV) |
    """)
    return


@app.cell
def _(bench_df, complete_df, mo, pl):
    _table_methods = ["graphsage", "rustgraph", "taddy", "sad", "addgraph", "strgnn"]

    # Average edges/s and GPU across all 6 datasets
    _avg_bench = (
        bench_df.filter(pl.col("method").is_in(_table_methods))
        .group_by("method")
        .agg(
            pl.mean("edges_per_sec").alias("avg_edges_per_sec"),
            pl.mean("peak_gpu_mb").alias("avg_gpu_mb"),
        )
    )

    # Inference time on as-topology (largest dataset)
    _inference_as = (
        bench_df.filter(
            pl.col("method").is_in(_table_methods),
            pl.col("dataset") == "as-topology",
        ).select("method", "inference_time_sec")
    )

    # Mean AUC across all conditions from the full CSV
    _mean_auc = (
        complete_df.filter(pl.col("method").is_in(_table_methods))
        .group_by("method")
        .agg(pl.mean("roc_auc").alias("mean_auc"))
    )

    scalability_table = (
        _avg_bench.join(_inference_as, on="method")
        .join(_mean_auc, on="method")
        .sort("avg_edges_per_sec", descending=True)
    )

    def _fmt_edges(v):
        if v >= 1000:
            return f"{v / 1000:.1f}K"
        return f"{v:.0f}"

    _rows = []
    for _row in scalability_table.iter_rows(named=True):
        _rows.append(
            f"| {_row['method']} | {_fmt_edges(_row['avg_edges_per_sec'])} "
            f"| {_row['avg_gpu_mb']:,.0f} "
            f"| {_row['inference_time_sec']:.2f} "
            f"| {_row['mean_auc']:.2f} |"
        )

    _table_md = (
        "| Method | Edges/s | GPU (MB) | Inference (s) | AUC |\n"
        "|--------|---------|----------|---------------|-----|\n"
        + "\n".join(_rows)
    )

    mo.md(f"**Computed Table (cf. Table in `scalability_section.tex`):**\n\n{_table_md}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top-4 Method AUC Range

    > "The top four methods … are separated by only 2 percentage points in mean AUC (0.74–0.76)"
    """)
    return


@app.cell
def _(complete_df, mo, pl):
    _top4 = ["strgnn", "sad", "rustgraph", "taddy"]

    _top4_auc = (
        complete_df.filter(pl.col("method").is_in(_top4))
        .group_by("method")
        .agg(pl.mean("roc_auc").alias("mean_auc"))
        .sort("mean_auc", descending=True)
    )

    _rows = "\n".join(
        f"| {r['method']} | {r['mean_auc']:.3f} |"
        for r in _top4_auc.iter_rows(named=True)
    )

    _auc_min = _top4_auc["mean_auc"].min()
    _auc_max = _top4_auc["mean_auc"].max()
    _spread_pp = (_auc_max - _auc_min) * 100

    mo.md(
        f"| Method | Mean AUC |\n|--------|----------|\n{_rows}\n\n"
        f"**Spread**: {_auc_max:.2f} – {_auc_min:.2f} = **{_spread_pp:.0f} pp**"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scaling Behavior: Train Time

    > "GraphSAGE increases from 0.3s to 2.9s (9.7×) when moving from 17K to 159K edges (a 9.2× increase),
    > and RustGraph from 1.9s to 11.4s (6.0×)"
    """)
    return


@app.cell
def _(bench_df, mo, pl):
    _smallest = "bitcoin-alpha"
    _largest = "as-topology"

    _edges_small = (
        bench_df.filter(pl.col("dataset") == _smallest)
        .select("num_edges")
        .row(0)[0]
    )
    _edges_large = (
        bench_df.filter(pl.col("dataset") == _largest)
        .select("num_edges")
        .row(0)[0]
    )
    _edge_ratio = _edges_large / _edges_small

    _lines = [
        f"**Edge ratio** ({_largest} / {_smallest}): "
        f"{_edges_large:,} / {_edges_small:,} = **{_edge_ratio:.2f}x**\n"
    ]

    for _method in ["graphsage", "rustgraph", "strgnn", "sad", "taddy", "addgraph"]:
        _mdata = bench_df.filter(
            pl.col("method") == _method,
            pl.col("dataset").is_in([_smallest, _largest]),
        )
        _small = _mdata.filter(pl.col("dataset") == _smallest).row(0, named=True)
        _large = _mdata.filter(pl.col("dataset") == _largest).row(0, named=True)

        _t_small = _small["train_time_sec"]
        _t_large = _large["train_time_sec"]
        _time_ratio = _t_large / _t_small

        _lines.append(
            f"**{_method}**: {_t_small:.1f}s → {_t_large:.1f}s (**{_time_ratio:.1f}x**)"
        )

    mo.md("#### Train time: smallest → largest dataset\n\n" + "\n\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Throughput Flatness (StrGNN and SAD)

    > "throughput remains nearly flat at ~300 and ~470 edges/s regardless of graph size"
    """)
    return


@app.cell
def _(bench_df, mo, pl):
    _lines = []
    for _method in ["strgnn", "sad"]:
        _eps = bench_df.filter(pl.col("method") == _method).select(
            "dataset", "edges_per_sec"
        )
        _avg = _eps["edges_per_sec"].mean()
        _min_v = _eps["edges_per_sec"].min()
        _max_v = _eps["edges_per_sec"].max()
        _lines.append(
            f"**{_method}**: avg **{_avg:.0f}** edges/s "
            f"(range {_min_v:.0f}–{_max_v:.0f})"
        )
    mo.md("\n\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Super-linear Scaling: TADDY RAM and AddGraph GPU

    > "TADDY's … RAM consumption to reach 129 GB on the largest dataset"
    >
    > "AddGraph's GPU memory grows from 361 MB to 28 GB (78×) over a 9× increase in edges"
    """)
    return


@app.cell
def _(bench_df, mo, pl):
    # TADDY RAM on as-topology
    _taddy_as = bench_df.filter(
        pl.col("method") == "taddy", pl.col("dataset") == "as-topology"
    ).row(0, named=True)
    _taddy_ram_gb = _taddy_as["peak_ram_mb"] / 1024

    # AddGraph GPU: bitcoin-alpha → as-topology
    _add_small = bench_df.filter(
        pl.col("method") == "addgraph", pl.col("dataset") == "bitcoin-alpha"
    ).row(0, named=True)
    _add_large = bench_df.filter(
        pl.col("method") == "addgraph", pl.col("dataset") == "as-topology"
    ).row(0, named=True)
    _add_gpu_ratio = _add_large["peak_gpu_mb"] / _add_small["peak_gpu_mb"]

    mo.md(
        f"**TADDY** RAM on as-topology: {_taddy_as['peak_ram_mb']:,.0f} MB = "
        f"**{_taddy_ram_gb:,.0f} GB**\n\n"
        f"**AddGraph** GPU: {_add_small['peak_gpu_mb']:,.0f} MB → "
        f"{_add_large['peak_gpu_mb']:,.0f} MB = **{_add_gpu_ratio:.0f}x**"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference Latency on as-topology (159K edges)

    > "GraphSAGE completes inference in 47 ms, TADDY in 42 ms, and RustGraph in 246 ms …
    > StrGNN and SAD require 38 s and 91 s respectively …
    > This creates a 1,930× latency gap"
    """)
    return


@app.cell
def _(bench_df, mo, pl):
    _as_inference = (
        bench_df.filter(pl.col("dataset") == "as-topology")
        .select("method", "inference_time_sec")
        .sort("inference_time_sec")
    )

    _rows = []
    for _row in _as_inference.iter_rows(named=True):
        _t = _row["inference_time_sec"]
        if _t < 1.0:
            _rows.append(f"| {_row['method']} | {_t * 1000:.0f} ms |")
        else:
            _rows.append(f"| {_row['method']} | {_t:.1f} s |")

    _fastest = _as_inference.row(0, named=True)
    _slowest = _as_inference.row(-1, named=True)
    _gap = _slowest["inference_time_sec"] / _fastest["inference_time_sec"]

    _table = (
        "| Method | Inference time |\n|--------|----------------|\n" + "\n".join(_rows)
    )

    mo.md(
        f"{_table}\n\n"
        f"**Latency gap**: {_slowest['method']} ({_slowest['inference_time_sec']:.1f}s) / "
        f"{_fastest['method']} ({_fastest['inference_time_sec'] * 1000:.0f}ms) = "
        f"**{_gap:,.0f}x**"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RustGraph: Structural–Temporal Tradeoff

    > "RustGraph … achieves competitive AUC … (0.75 overall, including the best performance
    > on burst anomalies at 0.87) … at 100× less training cost than StrGNN or SAD"
    """)
    return


@app.cell
def _(bench_df, complete_df, mo, pl):
    # RustGraph AUC per anomaly type
    _rg_auc = complete_df.filter(pl.col("method") == "rustgraph")
    _rg_overall = _rg_auc["roc_auc"].mean()

    _rg_per_type = (
        _rg_auc.group_by("anomaly_type")
        .agg(pl.mean("roc_auc").alias("mean_auc"))
        .sort("mean_auc", descending=True)
    )

    _rows = "\n".join(
        f"| {r['anomaly_type']} | {r['mean_auc']:.2f} |"
        for r in _rg_per_type.iter_rows(named=True)
    )

    # Training cost ratio: avg train_time of strgnn and sad vs rustgraph
    _avg_train = (
        bench_df.filter(pl.col("method").is_in(["rustgraph", "strgnn", "sad"]))
        .group_by("method")
        .agg(pl.mean("train_time_sec").alias("avg_train"))
    )
    _rg_train = _avg_train.filter(pl.col("method") == "rustgraph").row(0, named=True)[
        "avg_train"
    ]
    _strgnn_train = _avg_train.filter(pl.col("method") == "strgnn").row(0, named=True)[
        "avg_train"
    ]
    _sad_train = _avg_train.filter(pl.col("method") == "sad").row(0, named=True)[
        "avg_train"
    ]

    mo.md(
        f"**RustGraph overall mean AUC: {_rg_overall:.2f}**\n\n"
        f"| Anomaly type | Mean AUC |\n|--------------|----------|\n{_rows}\n\n"
        f"**Training cost ratios** (avg train time across datasets):\n\n"
        f"- RustGraph: {_rg_train:.1f}s\n"
        f"- StrGNN: {_strgnn_train:.1f}s → **{_strgnn_train / _rg_train:.0f}x** slower\n"
        f"- SAD: {_sad_train:.1f}s → **{_sad_train / _rg_train:.0f}x** slower"
    )
    return


if __name__ == "__main__":
    app.run()
