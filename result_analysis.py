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
        fig = plt.figure(figsize=(18, 8))

        subfigs_top = fig.subfigures(1, 3, wspace=0.15)
        axes_top = [subfig.subplots(1, 1) for subfig in subfigs_top]
    
        subfigs_bottom = fig.subfigures(1, 3, wspace=0.15)
    
        # Create figure-level grid specification for centering
        gs = fig.add_gridspec(2, 3)
    
        # Clear and use a simpler approach with gridspec
        plt.close("all")  # Close previous figure
    
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3)
    
        # Row 1: 3 plots (columns 0, 1, 2)
        axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
        # Row 2: 2 plots centered (columns 0 and 1)
        axes_row2 = [fig.add_subplot(gs[1, i]) for i in range(2)]
    
        axes = axes_row1 + axes_row2
    
        for idx, ax in enumerate(axes):
            fig_single = plt.figure(figsize=(5, 4))
            ax_single = fig_single.add_subplot(111)
    
            # Plot the same data
            an_type = anomaly_types[idx]
            df_filtered = complete_df.filter(
                pl.col("anomaly_type") == an_type, pl.col("method").is_in(method_names)
            )
    
            sns.lineplot(
                data=df_filtered,
                x="anomaly_rate",
                y="roc_auc",
                hue="method",  # You can use: dataset, duration, anomaly_rate, etc.
                palette=palette,  # For the hue variable, can be any seaborn palette
                ax=ax_single,
                errorbar="ci",
                marker="o",
                markersize=10
            )
    
            if idx == 0:  # Keep legend only on first plot
                ax_single.legend(title='Method', fontsize=14, title_fontsize=16)
            else:
                legend = ax_single.get_legend()
                if legend:
                    legend.remove()
    
            # Apply the same global y-limits
            ax_single.set_ylim(0, 1)
            ax_single.set_xlabel("$\mathcal{R}$", fontsize=14)
            ax_single.set_ylabel("ROC AUC", fontsize=14)
            ax_single.tick_params(axis='both', labelsize=12)
            # ax_single.set_title(
            #     f"{an_type}".capitalize(), fontsize=14, fontweight="bold"
            # )
    
            # NO label text added here either - LaTeX subcaption will handle it
    
            fig_single.savefig(
                f"paper_figures/subplot_{chr(97 + idx)}_clean.pdf", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_single)

    plot_exp_1()
    return


@app.cell
def _(complete_df, pl, sns):
    new_exp_1_data = (complete_df
                  .filter(pl.col("method").is_in(["strgnn", "taddy", "sad", "addgraph", "rustgraph"]))
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


@app.cell
def _(complete_df):
    complete_df
    return


@app.cell
def _(mo):
    experiment_2_visual = mo.ui.radio(["point", "bar", "box", "violin", "boxen", "strip", "swarm"])
    experiment_2_visual
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


@app.cell
def _(mo):
    experiment_4_visual = mo.ui.radio(["point", "bar", "box", "violin", "boxen", "strip", "swarm"])
    experiment_4_visual
    return (experiment_4_visual,)


@app.cell
def _(complete_df, experiment_4_visual, pl, sns):
    exp4_data = (complete_df
        .filter(pl.col("anomaly_rate") == 0.05)
        .filter(pl.col("duration") == 0.5)
        .filter(pl.col("anomaly_type").is_in(["path", "random", "bridge"]))
        .filter(pl.col("method").is_in(["strgnn", "taddy", "sad", "addgraph", "rustgraph"]))
                  # .group_by(["method", "anom_type", "anom_ratio", "anom_duration"]).agg(pl.mean("roc_auc"))
    )

    exp4_plot = sns.catplot(
        data=exp4_data,
        kind=f"{experiment_4_visual.value}",
        x="anomaly_type",
        y="roc_auc",
        hue="method",
        col="dataset"
    )


    exp4_plot.savefig("paper_figures/experiment_4_simple_anomalies", dpi=600)

    exp4_plot
    return


@app.cell
def _(complete_df, pl, sns):
    exp5_data = (complete_df
        .filter(pl.col("anomaly_rate") == 0.05)
        .filter(pl.col("duration") == 1.0)
        .filter(pl.col("anomaly_type").is_in(["burst", "clique"]))
        .filter(pl.col("method").is_in(["strgnn", "taddy", "sad", "addgraph", "rustgraph"]))
                  # .group_by(["method", "anom_type", "anom_ratio", "anom_duration"]).agg(pl.mean("roc_auc"))
    )

    exp5_plot = sns.catplot(
        data=exp5_data,
        kind="bar",
        x="anomaly_type",
        y="roc_auc",
        hue="method",
        col="dataset"
    )



    exp5_plot.savefig("paper_figures/experiment_5_complex_anomalies", dpi=600)

    exp5_plot
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
