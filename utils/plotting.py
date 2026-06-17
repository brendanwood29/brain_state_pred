import itertools
from pathlib import Path
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

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

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
    sns.boxplot(
        data=df,
        x="Model",
        y=variable,
        order=names,
    )
    annotator.annotate()
    plt.ylabel(f"{variable}")
    plt.tight_layout()
    plt.savefig(f"{fname}.svg")
    plt.close()


if __name__ == "__main__":
    mpl.rcParams["axes.labelsize"] = 20  # x and y axis labels
    mpl.rcParams["xtick.labelsize"] = 18  # x tick labels
    mpl.rcParams["ytick.labelsize"] = 18  # y tick labels
    mpl.rcParams["axes.titlesize"] = 24  # plot title
    mpl.rcParams["font.size"] = 20  # base font size
    my_colors = ["#DC4633"]
    sns.set_theme(font_scale=2.25)
    sns.set_palette(my_colors)

    baseline = pd.read_csv("results/baseline_new_set/baseline_new_set.csv")
    baseline.insert(0, "Model", baseline.shape[0] * ["Baseline Single Sub."])
    baseline = baseline[baseline["recon_err"] < 8]
    mamba_ss = pd.read_csv(
        "results/mamba_ss_hparam/dropout-0.3_k-117_lr-0.0005/dropout-0.3_k-117_lr-0.0005.csv"
    )
    mamba_ss.insert(0, "Model", mamba_ss.shape[0] * ["Mamba Single Sub."])
    mamba_ptft = pd.read_csv("results/mamba_400p/mamba_400p.csv")
    mamba_ptft.insert(0, "Model", mamba_ptft.shape[0] * ["Mamba PT/FT"])
    models = {
        "Mamba Single Sub.": mamba_ss,
        "Mamba PT/FT": mamba_ptft,
        # "Transformer Single Subject": None,
        # "Transformer Pretrained/Fine-Tuned": None,
        "Baseline Single Sub.": baseline,
    }

    boxplot_with_stars(
        models,
        "r",
        pg.rm_anova,
        {
            "dv": "r",
            "within": "Model",
            "subject": "Scan",
            "correction": True,
            "detailed": True,
        },
        pg.pairwise_tests,
        {"dv": "r", "within": "Model", "subject": "Scan", "padjust": "fdr_bh"},
        fname="r",
    )

    boxplot_with_stars(
        models,
        "recon_err",
        pg.rm_anova,
        {
            "dv": "recon_err",
            "within": "Model",
            "subject": "Scan",
            "correction": True,
            "detailed": True,
        },
        pg.pairwise_tests,
        {"dv": "recon_err", "within": "Model", "subject": "Scan", "padjust": "fdr_bh"},
        fname="recon",
    )
    # transformer_dir = Path("results/symposium/single_subject-1")
    # baseline_dir = Path("results/symposium/baseline-1")
    # trans_all_dir = Path("results/symposium/fine_tuned-1")
    # fig_dir = Path("results/symposium/figures")
    # fig_dir.mkdir(parents=True, exist_ok=True)
    #
    # with open("100p_labels.txt", "r") as f:
    #     labels = f.readlines()
    # labels = [x.removesuffix("\n") for x in labels]
    #
    # df = pd.read_csv("results/symposium/all_models.csv", index_col=0)
    #
    # for sub in trans_all_dir.iterdir():
    #     sub_df = df[df["Scan"] == sub.name]
    #     print(sub_df)
    #     sub_name = sub.name
    #     real_fc = pd.read_csv(sub.joinpath("connectomes", "real.csv"), index_col=0)
    #     trans_ss_fc = pd.read_csv(
    #         transformer_dir.joinpath(sub_name, "connectomes", "model.csv"), index_col=0
    #     )
    #     baseline_ss_fc = pd.read_csv(
    #         baseline_dir.joinpath(sub_name, "connectomes", "model.csv"), index_col=0
    #     )
    #     trans_all_fc = pd.read_csv(
    #         sub.joinpath("connectomes", "model.csv"), index_col=0
    #     )
    #     plot_fcs(
    #         [real_fc, trans_ss_fc, baseline_ss_fc, trans_all_fc],
    #         [
    #             "Real FC",
    #             "Transformer Single Subject FC",
    #             "Baseline FC",
    #             "Transformer Fine-Tuned FC",
    #         ],
    #         fig_dir.joinpath(sub_name + ".svg"),
    #     )
