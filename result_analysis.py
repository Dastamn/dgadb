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
    return pl, sns


@app.cell
def _(pl):
    complete_version_path  = "MASTER_roc_auc_subsettest_final_final_long_form.csv"
    complete_df = pl.read_csv(complete_version_path)
    return (complete_df,)


@app.cell
def _(complete_df):
    complete_df
    return


@app.cell
def _():
    import polars.selectors as cs
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


@app.cell
def _():
 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 1
    """)
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

    for ax, title in zip(exp4_plot.axes.flat, [f"caption {i}" for i in range(6)]):
        ax.text(
            0.5, 0.02,                 # x, y in Axes coords (0-1)
            title,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9, style="italic"
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
