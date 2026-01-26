import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # New version
    This version generates the master table without pivot.
    This should make it easier to plot and analyse.
    """)
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    file_list = [
    "addgraph-bitcoin-alpha.csv",
    "rustgraph-as-topology.csv",
    "rustgraph-bitcoin-alpha-email-dnc-dur-1.csv",
    "rustgraph-bitcoin-otc-uc-social-rate-0.05-dur-1.csv",
    "rustgraph-digg.csv",
    "rustgraph-uc-social-rate-0.05-dur-1.csv",
    "sad-as-topology.csv",
    "sad-bitcoin-uc-social-email-dnc-dur-1.csv",
    "sad-bitcoin-uc-social-email-dnc-rate-0.05 .csv",
    "sad-digg.csv",
    "sad-email-dnc-rate-0.01-0.05-WEIRD.csv",
    "slade-all-small-datasets.csv",
    "strgnn-as-topology.csv",
    "strgnn-bitcoin-alpha-dur-1.csv",
    "strgnn-digg.csv",
    "strgnn-email-dnc-rate-0.01.csv",
    "strgnn-email-dnc-rate-0.1-dur-1.csv",
    "strgnn-uc-social.csv",
    "taddy-as-topology.csv",
    "taddy-bitcoin-dur-1.csv",
    "taddy-digg.csv",
    "taddy-uc-social-email-dnc.csv",
    ]
    return (file_list,)


@app.cell
def _():
    from pathlib import Path
    import os
    import numpy as np

    os.environ["BASE_PATH"] = "."
    from dgadb.preprocessing.anomaly_injection import _ANOMALY_DURATION_TYPE_MAP
    dir_base = Path("aim_output/")
    return Path, dir_base, np


@app.cell
def _(Path, pd):
    def process_csv(path: str, dir_base: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(dir_base / path)

            # Clean "run" column first to avoid issues
            df = df.dropna(subset=['run'])

            # 1. Extract Method and Dataset
            df['method'] = df['run'].str.split('_').str[0].str.replace('"', '').str.strip()
            df['dataset'] = df['run'].str.split('_').str[1].str.replace('"', '').str.strip()

            # 2. Clean Attribute names
            df = df.rename(columns=lambda x: x.replace('anom_config.', ''))

            # 3. MAP DURATIONS (Robust handling for small/medium/large)
            if 'anom_duration' in df.columns:
                # Convert to string, clean quotes, map, then convert back to numeric
                df['anom_duration'] = df['anom_duration'].astype(str).str.replace('"', '').str.strip().str.lower()
                df['anom_duration'] = df['anom_duration'].map(_ANOMALY_DURATION_TYPE_MAP).fillna(df['anom_duration'])

            # 4. Numeric conversion
            df['anom_ratio'] = pd.to_numeric(df['anom_ratio'], errors='coerce')
            df['anom_duration'] = pd.to_numeric(df['anom_duration'], errors='coerce')

            # 5. Parse date
            df['date_dt'] = pd.to_datetime(df['date'].str.replace(' · ', ' '), format='%H:%M:%S %d %b, %y', errors='coerce')

            return df
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")
    return (process_csv,)


@app.cell
def _(dir_base, file_list, process_csv):
    all_data = []
    for f in file_list:
        all_data.append(process_csv(f, dir_base))
    return (all_data,)


@app.cell
def _(all_data, np, pd):
    full_df = pd.concat(all_data)
    full_df = full_df.dropna(subset="dataset")
    dedup_keys = ['method', 'dataset', 'anom_type', 'anom_ratio', 'anom_duration']
    full_df = full_df.drop_duplicates(subset=dedup_keys, keep='last')
    meta_cols = ['run', 'date', 'duration', 'tags', 'date_dt', 'dataset', 'method', 
                 'anom_type', 'anom_ratio', 'anom_duration']
    model_cols = [c for c in full_df.columns if 'model.components' in c]
    metrics = [c for c in full_df.columns if c not in meta_cols + model_cols]

    metrics_obj_cols = [
        'roc_auc subset="test" ',
        'average_precision subset="test" ',
        'balanced_accuracy subset="test" ',
        'best_threshold subset="test" ',
        'max_f1 subset="test" '
    ]

    other_obj_cols = [
        'dataset',
        'method',
    ]

    single_metric = ['roc_auc subset="test" '] 
    relevant_meta = [
        "dataset",
        "method",
        "anom_type",
        "anom_ratio",
        "anom_duration"
    ]

    full_df[metrics_obj_cols] = full_df[metrics_obj_cols].replace("-", np.nan)
    full_df[metrics_obj_cols] = full_df[metrics_obj_cols].astype(float)
    full_df[other_obj_cols] = full_df[other_obj_cols].astype(str)
    full_df = full_df[single_metric + relevant_meta]
    return (full_df,)


@app.cell
def _(full_df, pl):
    full_df_pl = pl.from_dataframe(full_df).drop("__index_level_0__")
    full_df_pl = full_df_pl.with_columns(pl.col("anom_type").str.strip_chars('"'))
    full_df_pl = full_df_pl.rename(
        {'roc_auc subset="test" ': "roc_auc"}
    )
    # full_df_pl = full_df_pl.filter(!pl.col("method") == "gene")
    full_df_pl = full_df_pl.drop_nulls()
    return (full_df_pl,)


@app.cell
def _(full_df_pl):
    full_df_pl
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment 1: frequency analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The first experiment aim at discarding frequency at eliminating one of the dimensions of the analysis.
    """)
    return


@app.function
def increase_figure_font_size(plot):
    for ax in plot.axes.flat:
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt
    return (sns,)


@app.cell
def _(full_df_pl):
    full_df_pl
    return


@app.cell
def _(full_df_pl, pl, sns):
    exp_1_data = (full_df_pl.group_by(["method", "anom_type", "anom_ratio", "anom_duration"]).agg(pl.mean("roc_auc"))
                  .filter(pl.col("method").is_in(["strgnn", "taddy", "sad", "addgraph", "rustgraph"]))
                 )

    exp_1_plot = sns.relplot(
        data=exp_1_data,
        kind="line",
        x="anom_ratio",
        y="roc_auc",
        col="anom_type",
        hue="method",
        row="anom_duration",
        marker="o",
        markersize=10
    )

    increase_figure_font_size(exp_1_plot)

    exp_1_plot.savefig("paper_figures/experiment_1_frequency")

    exp_1_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment 2: Temporal span
    """)
    return


@app.cell
def _(full_df_pl, pl, sns):
    exp_2_data = (full_df_pl
        .filter(pl.col("anom_ratio") == 0.05)
        .filter(pl.col("method").is_in(["strgnn", "taddy", "sad", "addgraph", "rustgraph"]))
        .filter(pl.col("anom_duration").is_in([0.1, 1.0]))
                  # .group_by(["method", "anom_type", "anom_ratio", "anom_duration"]).agg(pl.mean("roc_auc"))
    )

    exp_2_plot = sns.catplot(
        data=exp_2_data,
        kind="box",
        x="anom_duration",
        y="roc_auc",
        hue="method",
        col="anom_type"
    )

    increase_figure_font_size(exp_2_plot)


    exp_2_plot.savefig("paper_figures/experiment_2_temporal_span")

    exp_2_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment 4: method performance on simple types
    """)
    return


@app.cell
def _(exp_4_data):
    exp_4_data
    return


@app.cell
def _(full_df_pl, pl):
    (full_df_pl.filter(pl.col("anom_duration") == 0.5, pl.col("anom_ratio") == 0.05)
     .group_by(["method", "dataset", "anom_type"]).agg(pl.count("roc_auc"))
    )
    return


@app.cell
def _(full_df_pl, pl, sns):
    exp_4_data = (full_df_pl
        .filter(pl.col("anom_ratio") == 0.05)
        .filter(pl.col("anom_type").is_in(["path", "random", "bridge"]))
        .filter(pl.col("method").is_in(["strgnn", "taddy", "sad", "addgraph", "rustgraph"]))
                  # .group_by(["method", "anom_type", "anom_ratio", "anom_duration"]).agg(pl.mean("roc_auc"))
    )

    exp_4_plot = sns.catplot(
        data=exp_4_data,
        kind="bar",
        x="anom_type",
        y="roc_auc",
        hue="method",
        col="dataset"
    )

    increase_figure_font_size(exp_4_plot)


    exp_4_plot.savefig("paper_figures/experiment_4_simple_anomalies")

    exp_4_plot
    return (exp_4_data,)


@app.cell
def _(full_df_pl, pl):
    selected_df = (full_df_pl
        .filter(pl.col("anom_type").is_in(["clique", "burst"]))
                   .filter(pl.col("anom_ratio") == 0.05)
                  )
    selected_df
    return (selected_df,)


@app.cell
def _(selected_df, sns):
    g = sns.catplot(
        data=selected_df,
        kind="box",
        x="anom_type",
        y="roc_auc",
        hue="method",
        col="dataset"
    )

    g.savefig("nazim_nice_figure")
    return (g,)


@app.cell
def _(g):
    g
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Old version
    This version tried to recreate the master table without the pivots.
    Since I now have access to the full tables, I can reconstruct it with Nazim's code.
    """)
    return


@app.cell
def _():
    source_file = "MASTER_roc_auc_subsettest_final.xlsx"
    return (source_file,)


@app.cell
def _(pl, source_file):
    full_spreadsheet = pl.read_excel(source_file, has_header=False)
    return (full_spreadsheet,)


@app.cell
def _(full_spreadsheet):
    full_spreadsheet
    return


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl, source_file):
    hdr = (pl.read_excel(source_file, has_header=False, read_options={"n_rows": 5})
           .transpose()
           .drop("column_4")
           .rename(
               {
                   "column_0": "dataset",
                   "column_1": "type",
                   "column_2": "rate",
                   "column_3": "temporal_span"
               }
           )
          )
    return (hdr,)


@app.cell
def _(hdr):
    cleaned_headers = hdr.fill_null(strategy="forward").drop_nulls()
    return (cleaned_headers,)


@app.cell
def _(cleaned_headers):
    cleaned_headers
    return


@app.cell
def _(pl, source_file):
    df = pl.read_excel(source_file, has_header=False, read_options={"skip_rows": 5}).transpose()
    return (df,)


@app.cell
def _(df):
    df.row(0)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Cleaned version

    The csv file is now in long form.
    """)
    return


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

    increase_figure_font_size(new_exp_1_plot)

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

    increase_figure_font_size(new_exp_2_plot)


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

    increase_figure_font_size(exp4_plot)

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

    increase_figure_font_size(exp5_plot)


    exp5_plot.savefig("paper_figures/experiment_5_complex_anomalies", dpi=600)

    exp5_plot
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
