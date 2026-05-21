from pathlib import Path

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


if __name__ == "__main__":
    mpl.rcParams["axes.labelsize"] = 20  # x and y axis labels
    mpl.rcParams["xtick.labelsize"] = 18  # x tick labels
    mpl.rcParams["ytick.labelsize"] = 18  # y tick labels
    mpl.rcParams["axes.titlesize"] = 24  # plot title
    mpl.rcParams["font.size"] = 20  # base font size

    df = pd.read_csv("results/symposium/single_subject-1.csv")
    df.insert(0, "Model", ["Single Subject"] * df.shape[0])
    df1 = pd.read_csv("results/symposium/baseline-1.csv")
    df1.insert(0, "Model", ["Baseline"] * df1.shape[0])
    df2 = pd.read_csv("results/symposium/fine_tuned-1.csv")
    df2.insert(0, "Model", "Pretrained/Fine-tuned")

    subs = df2["Scan"].tolist()

    df = pd.concat([df, df1, df2], ignore_index=True)
    df = df[df["Scan"].isin(subs)]
    df.to_csv("all_models.csv")

    res = pg.rm_anova(
        data=df, dv="r", within="Model", subject="Scan", correction=True, detailed=True
    )
    post = pg.pairwise_tests(
        data=df, dv="r", within="Model", subject="Scan", padjust="fdr_bh"
    )
    print(post.to_string())
    pairs = [
        ("Baseline", "Single Subject"),
        ("Baseline", "Pretrained/Fine-tuned"),
        ("Single Subject", "Pretrained/Fine-tuned"),
    ]
    pvalues = [
        post.loc[
            (post.A == "Baseline") & (post.B == "Single Subject"), "p_corr"
        ].values[0],
        post.loc[
            (post.A == "Baseline") & (post.B == "Pretrained/Fine-tuned"), "p_corr"
        ].values[0],
        post.loc[
            (post.B == "Single Subject") & (post.A == "Pretrained/Fine-tuned"), "p_corr"
        ].values[0],
    ]
    my_colors = ["#DC4633"]

    # Set it globally
    sns.set_theme(font_scale=2.25)
    sns.set_palette(my_colors)
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    annotator = Annotator(
        ax,
        pairs,
        data=df,
        x="Model",
        y="r",
        order=["Baseline", "Single Subject", "Pretrained/Fine-tuned"],
    )
    annotator.configure(hide_non_significant=True, text_format="simple", loc="outside")
    annotator.set_pvalues(pvalues)
    annotator.annotate()
    sns.boxplot(
        data=df,
        x="Model",
        y="r",
        order=["Baseline", "Single Subject", "Pretrained/Fine-tuned"],
    )
    plt.ylabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    plt.savefig("pcc.svg")
    plt.close()
    df = df[df["recon_err"] < 10]

    res = pg.rm_anova(
        data=df,
        dv="recon_err",
        within="Model",
        subject="Scan",
        correction=True,
        detailed=True,
    )
    post = pg.pairwise_tests(
        data=df, dv="recon_err", within="Model", subject="Scan", padjust="fdr_bh"
    )
    print(post.to_string())
    pairs = [
        ("Baseline", "Single Subject"),
        ("Baseline", "Pretrained/Fine-tuned"),
        ("Single Subject", "Pretrained/Fine-tuned"),
    ]
    pvalues = [
        post.loc[
            (post.A == "Baseline") & (post.B == "Single Subject"), "p_corr"
        ].values[0],
        post.loc[
            (post.A == "Baseline") & (post.B == "Pretrained/Fine-tuned"), "p_corr"
        ].values[0],
        post.loc[
            (post.B == "Single Subject") & (post.A == "Pretrained/Fine-tuned"), "p_corr"
        ].values[0],
    ]
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    annotator = Annotator(
        ax,
        pairs,
        data=df,
        x="Model",
        y="recon_err",
        order=["Baseline", "Single Subject", "Pretrained/Fine-tuned"],
    )
    annotator.configure(hide_non_significant=True, text_format="simple", loc="outside")
    annotator.set_pvalues(pvalues)
    sns.boxplot(
        data=df,
        x="Model",
        y="recon_err",
        order=["Baseline", "Single Subject", "Pretrained/Fine-tuned"],
        ax=ax,
    )
    plt.ylabel("Mean Squared Reconstruction Error")
    annotator.annotate()
    plt.tight_layout()
    plt.savefig("recon.svg")
    plt.close()

    transformer_dir = Path("results/symposium/single_subject-1")
    baseline_dir = Path("results/symposium/baseline-1")
    trans_all_dir = Path("results/symposium/fine_tuned-1")
    fig_dir = Path("results/symposium/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open("100p_labels.txt", "r") as f:
        labels = f.readlines()
    labels = [x.removesuffix("\n") for x in labels]

    df = pd.read_csv("results/symposium/all_models.csv", index_col=0)

    for sub in trans_all_dir.iterdir():
        sub_df = df[df["Scan"] == sub.name]
        print(sub_df)
        sub_name = sub.name
        real_fc = pd.read_csv(sub.joinpath("connectomes", "real.csv"), index_col=0)
        trans_ss_fc = pd.read_csv(
            transformer_dir.joinpath(sub_name, "connectomes", "model.csv"), index_col=0
        )
        baseline_ss_fc = pd.read_csv(
            baseline_dir.joinpath(sub_name, "connectomes", "model.csv"), index_col=0
        )
        trans_all_fc = pd.read_csv(
            sub.joinpath("connectomes", "model.csv"), index_col=0
        )
        plot_fcs(
            [real_fc, trans_ss_fc, baseline_ss_fc, trans_all_fc],
            [
                "Real FC",
                "Transformer Single Subject FC",
                "Baseline FC",
                "Transformer Fine-Tuned FC",
            ],
            fig_dir.joinpath(sub_name + ".svg"),
        )
