import itertools
from pathlib import Path
from typing import Callable, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from numpy import save
from statannotations.Annotator import Annotator


def boxplot_with_stars(
    models: Dict[str, pd.DataFrame],
    variable: str,
    stat_test: Callable[..., pd.DataFrame],
    stat_test_kwargs: dict,
    ph_test: Callable[..., pd.DataFrame],
    ph_kwargs: dict,
    save_path: Path,
    fname: str,
):
    # Make figure object
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    names = list(models.keys())
    names.sort()
    comparisons = list(itertools.combinations(names, 2))
    df = pd.concat([x for x in models.values()], ignore_index=True)
    result = stat_test(df, **stat_test_kwargs)

    # Only plot with stars if significant
    if result["p_unc"].iloc[0] < 0.05:
        p_vals = []
        result = ph_test(df, **ph_kwargs)
        for comparison in comparisons:
            p_vals.append(
                result.loc[
                    (result.A == comparison[0]) & (result.B == comparison[1]),
                    "p_corr",
                ].values[0]
            )
        annotator = Annotator(
            ax,
            comparisons,
            data=df,
            x="Model",
            y=variable,
            order=names,
        )
        annotator.configure(
            hide_non_significant=True, text_format="simple", loc="outside"
        )
        annotator.set_pvalues(p_vals)
        sns.boxplot(data=df, x="Model", y=variable, order=names, ax=ax)
        annotator.annotate()
    else:
        sns.boxplot(data=df, x="Model", y=variable, order=names, ax=ax)

    plt.ylabel(f"{variable}")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path.joinpath(f"{fname}.svg"))
    plt.close()


def main():

    model_paths = {
        "baseline": "results/baseline_val_set/baseline_val_set.csv",
        "mamba": "results/block_state-(4, 32)_d_conv-2_expand-4/block_state-(4, 32)_d_conv-2_expand-4.csv",
        "transformer": "results/transformer_val_set/transformer_val_set.csv",
        "mamba_recursive": "results/recursive_with_fc/recursive_with_fc.csv",
    }
    metrics = ["r", "psd_r"]
    save_path = Path("./figs/plotting_test")

    stat_test_kwargs = {
        "within": "Model",
        "subject": "Scan",
        "correction": True,
        "detailed": True,
    }
    ph_kwargs = {"within": "Model", "subject": "Scan", "padjust": "fdr_bh"}

    models = {}
    for model, path in model_paths.items():
        df = pd.read_csv(path)
        df.insert(df.shape[1], "Model", [model] * df.shape[0])
        models[model] = df

    for metric in metrics:
        stat_test_kwargs["dv"] = metric
        ph_kwargs["dv"] = metric
        boxplot_with_stars(
            models=models,
            variable=metric,
            stat_test=pg.rm_anova,
            stat_test_kwargs=stat_test_kwargs,
            ph_test=pg.pairwise_tests,
            ph_kwargs=ph_kwargs,
            save_path=save_path,
            fname=metric,
        )


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

    main()
