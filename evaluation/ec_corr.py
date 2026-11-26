import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def remove_diag(arr):
    return arr[~np.eye(arr.shape[0]).astype(bool)].flatten()


def plot_ec_diff(
    real_fc: np.ndarray, model_fc: np.ndarray, save_path: Path = Path(".")
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(28, 8),
    )
    ax1, ax2, ax3 = axes

    heatmap_kwargs = dict(
        cmap="RdBu_r",
        center=0,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )

    heatmap_kwargs["vmin"] = real_fc.min()
    heatmap_kwargs["vmax"] = real_fc.max()
    real_hm = sns.heatmap(real_fc, ax=ax1, **heatmap_kwargs)
    fig.colorbar(real_hm.collections[0], ax=ax1, shrink=0.6, pad=0.04)

    heatmap_kwargs["vmin"] = model_fc.min()
    heatmap_kwargs["vmax"] = model_fc.max()
    model_hm = sns.heatmap(model_fc, ax=ax2, **heatmap_kwargs)
    fig.colorbar(model_hm.collections[0], ax=ax2, shrink=0.6, pad=0.04)

    diff = model_fc - real_fc
    heatmap_kwargs["vmin"] = diff.min()
    heatmap_kwargs["vmax"] = diff.max()
    diff_hm = sns.heatmap(diff, ax=ax3, **heatmap_kwargs)
    fig.colorbar(diff_hm.collections[0], ax=ax3, shrink=0.6, pad=0.04)

    ax1.set_title("Real EC")
    ax2.set_title("Model EC")
    ax3.set_title("Model - Real EC")

    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path.joinpath("ec_heatmap.png"), bbox_inches="tight")
    plt.close()


def process_ec(arr: np.ndarray, nan_fill: float = 0):

    arr = np.nan_to_num(arr, nan=nan_fill)
    np.fill_diagonal(arr, 0)
    print(arr)
    arr /= np.abs(arr).max()

    return arr


def main():

    real_ec_path = ""
    model_ec_path = ""
    save_path = Path("./figs/test")

    real_ec = pd.read_csv(
        real_ec_path,
        index_col=0,
    ).to_numpy()
    model_ec = pd.read_csv(model_ec_path).to_numpy()

    real_ec = process_ec(real_ec)
    model_ec = process_ec(model_ec)

    plot_ec_diff(real_ec, model_ec, save_path)


if __name__ == "__main__":
    main()
