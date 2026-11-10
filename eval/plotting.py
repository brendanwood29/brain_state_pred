import itertools
from typing import Callable, Dict

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from statannotations.Annotator import Annotator


def plot_fcs(fcs, names, save_path):

    fig, axes = plt.subplots(2, 2, figsize=(25, 25))
    axes = axes.flatten()

    for i, (ax, fc, name) in enumerate(zip(axes, fcs, names)):
        fc = fc.to_numpy()

        sns.heatmap(
            fc,
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
            cbar=False,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )

        ax.set_title(name, fontsize=50)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # type: ignore

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    sm = cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label("Pearson Correlation Coefficient", fontsize=30, labelpad=0)

    plt.subplots_adjust(right=0.9)
    # plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, bbox_inches="tight")


def boxplot_with_stars(
    models: Dict[str, pd.DataFrame],
    variable: str,
    stat_test: Callable,
    stat_test_kwargs: dict,
    ph_test: Callable,
    ph_kwargs: dict,
    fname: str,
):
    names = list(models.keys())
    names.sort()
    comparisons = list(itertools.combinations(names, 2))
    df = pd.concat([x for x in models.values()], ignore_index=True)
    result = stat_test(df, **stat_test_kwargs)
    p_vals = []
    result = ph_test(df, **ph_kwargs)
    for comparison in comparisons:
        p_vals.append(
            result.loc[
                (result.A == comparison[0]) & (result.B == comparison[1]),
                "p_corr",
            ].values[0]
        )

    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    annotator = Annotator(
        ax,
        comparisons,
        data=df,
        x="Model",
        y=variable,
        order=names,
    )
    annotator.configure(hide_non_significant=True, text_format="simple", loc="outside")
    annotator.set_pvalues(p_vals)
    sns.boxplot(data=df, x="Model", y=variable, order=names, ax=ax)
    annotator.annotate()
    plt.ylabel(f"{variable}")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(f"{fname}.svg")
    plt.close()


if __name__ == "__main__":
    # This was good for a poster
    # mpl.rcParams["axes.labelsize"] = 20  # x and y axis labels
    # mpl.rcParams["xtick.labelsize"] = 18  # x tick labels
    # mpl.rcParams["ytick.labelsize"] = 18  # y tick labels
    # mpl.rcParams["axes.titlesize"] = 24  # plot title
    # mpl.rcParams["font.size"] = 20  # base font size
    # my_colors = ["#DC4633"]
    # sns.set_theme(font_scale=2.25)
    # sns.set_palette(my_colors)

    mpl.rcParams["axes.labelsize"] = 15  # x and y axis labels
    mpl.rcParams["xtick.labelsize"] = 13  # x tick labels
    mpl.rcParams["ytick.labelsize"] = 13  # y tick labels
    # mpl.rcParams["axes.titlesize"] = 24  # plot title
    mpl.rcParams["font.size"] = 15  # base font size
    my_colors = ["#DC4633"]
    sns.set_theme(font_scale=1.5)
    sns.set_palette(my_colors)

    # load models
    baseline = pd.read_csv(
        # "results/baseline_val_set_their-eval/baseline_val_set_their-eval.csv"
        "results/baseline_val_set/baseline_val_set.csv"
    )
    baseline.insert(0, "Model", baseline.shape[0] * ["Baseline Single Sub."])
    # baseline = baseline[baseline["recon_err"] < 40]
    mamba_ss = pd.read_csv(
        # "results/block_state-(4, 32)_d_conv-2_expand-4_their-eval/block_state-(4, 32)_d_conv-2_expand-4_their-eval.csv"
        "results/block_state-(4, 32)_d_conv-2_expand-4/block_state-(4, 32)_d_conv-2_expand-4.csv"
    )
    mamba_ss.insert(0, "Model", mamba_ss.shape[0] * ["Mamba Single Sub."])
    trans = pd.read_csv(
        # "results/transformer_val_set_their-eval/transformer_val_set_their-eval.csv"
        "results/transformer_val_set/transformer_val_set.csv"
    )
    trans.insert(0, "Model", trans.shape[0] * ["Transformer Single Sub."])
    models = {
        "Mamba Single Sub.": mamba_ss,
        "Transformer Single Sub.": trans,
        "Baseline Single Sub.": baseline,
    }

    # r
    dv = "r"
    boxplot_with_stars(
        models,
        dv,
        pg.rm_anova,
        {
            "dv": dv,
            "within": "Model",
            "subject": "Scan",
            "correction": True,
            "detailed": True,
        },
        pg.pairwise_tests,
        {"dv": dv, "within": "Model", "subject": "Scan", "padjust": "fdr_bh"},
        fname=dv,
    )

    # dtw
    dv = "dtw_dist"
    boxplot_with_stars(
        models,
        dv,
        pg.rm_anova,
        {
            "dv": dv,
            "within": "Model",
            "subject": "Scan",
            "correction": True,
            "detailed": True,
        },
        pg.pairwise_tests,
        {"dv": dv, "within": "Model", "subject": "Scan", "padjust": "fdr_bh"},
        fname=dv,
    )

    # recon err
    dv = "recon_err"
    boxplot_with_stars(
        models,
        dv,
        pg.rm_anova,
        {
            "dv": dv,
            "within": "Model",
            "subject": "Scan",
            "correction": True,
            "detailed": True,
        },
        pg.pairwise_tests,
        {"dv": dv, "within": "Model", "subject": "Scan", "padjust": "fdr_bh"},
        fname=dv,
    )

    # recon r
    dv = "recon_r"
    boxplot_with_stars(
        models,
        dv,
        pg.rm_anova,
        {
            "dv": dv,
            "within": "Model",
            "subject": "Scan",
            "correction": True,
            "detailed": True,
        },
        pg.pairwise_tests,
        {"dv": dv, "within": "Model", "subject": "Scan", "padjust": "fdr_bh"},
        fname=dv,
    )
    # psd
    dv = "psd_r"
    boxplot_with_stars(
        models,
        dv,
        pg.rm_anova,
        {
            "dv": dv,
            "within": "Model",
            "subject": "Scan",
            "correction": True,
            "detailed": True,
        },
        pg.pairwise_tests,
        {"dv": dv, "within": "Model", "subject": "Scan", "padjust": "fdr_bh"},
        fname=dv,
    )
