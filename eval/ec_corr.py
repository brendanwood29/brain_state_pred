import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_fc_diff(real_fc, model_fc, run):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(28, 8),
    )
    ax1, ax2, ax3 = axes

    heatmap_kwargs = dict(
        # vmin=-1.0,
        # vmax=1.0,
        cmap="RdBu_r",
        center=0,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )

    sns.heatmap(real_fc, ax=ax1, **heatmap_kwargs)  # type: ignore
    hm = sns.heatmap(model_fc, ax=ax2, **heatmap_kwargs)  # type: ignore
    sns.heatmap(real_fc - model_fc, ax=ax3, **heatmap_kwargs)  # type: ignore

    fig.colorbar(hm.collections[0], ax=[ax1, ax2, ax3], shrink=0.6, pad=0.02)
    ax1.set_title("Empirical FC")
    ax2.set_title("Model FC")
    ax3.set_title("Empirical - Model FC")

    plt.savefig("fc_heatmap.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    with open("splits/splits_66p/train.json", "r") as f:
        data = json.load(f)

    for item in data:
        data_path = (
            Path("data/data_66p/wbm_sim")
            .joinpath(item, "ses-1", "func")
            .joinpath(f"{item}_ses-1_EC.csv")
        )

        real_ec = pd.read_csv(data_path, index_col=0).to_numpy()
        real_ec[np.eye(real_ec.shape[0]).astype(bool)] = 0
        real_ec /= real_ec.max()

        results_path = Path(f"results/wbm_sim_naive_new/{item}_ses-1/ecs/ec.csv")
        model_ec = pd.read_csv(results_path, index_col=0).to_numpy()
        model_ec[np.eye(model_ec.shape[0]).astype(bool)] = 0
        model_ec /= model_ec.max()

        plt.figure()
        plt.scatter(real_ec.flatten(), model_ec.flatten())
        plt.savefig("test.svg")
        plot_fc_diff(real_ec, model_ec, run=None)
        import sys

        sys.exit()
