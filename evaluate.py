import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from nilearn.connectome import ConnectivityMeasure
from omegaconf import OmegaConf
from scipy.stats import pearsonr

from models import npi_model_getter as get_model
from utils import split_single_subject

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from tqdm import tqdm


def remove_diag(arr):

    return arr[~np.eye(arr.shape[0], dtype=bool)]


def sort_fn(path: Path):
    loss = path.name.split("_")[-1].removesuffix(".pt")
    return float(loss)


def find_best_loss(path):
    model = path.joinpath("models/final_model.pt")
    return model
    # models = [x for x in path.rglob('**/*best_val_loss*.pt')]
    # return sorted(models, key=sort_fn)[0]


def find_raw_data(path, name, suff="timeseries"):
    return list(path.rglob(f"**/{name}*{suff}.csv"))[0]


def get_recon(model, real_data, steps):
    device = next(model.parameters()).device
    model.eval()
    outputs = [torch.tensor(real_data[x]) for x in range(steps)]
    # for _ in range(steps, real_data.shape[0]):
    while len(outputs) < real_data.shape[0] - 20:
        # try:
        sim_input = torch.vstack([x.to(device) for x in outputs[-steps:]]).to(
            device=device, dtype=torch.float
        )
        outputs.extend([x for x in model(sim_input.unsqueeze(0).to(device)).squeeze(0)])
        outputs = [x.to("cpu") for x in outputs]
        # except:
        # sim_input = torch.cat(outputs[-steps:]).to(device=device, dtype=torch.float)
        # outputs.append(
        #     model(sim_input.unsqueeze(0).to(device)).squeeze(0)
        # )
    outputs = torch.vstack(outputs).detach().cpu().numpy()

    return outputs, np.mean(
        (real_data[steps + 1 : outputs.shape[0]] - outputs[steps + 1 :]) ** 2
    )


def get_model_fc(model, steps, sim_data_length, num_regions):

    rng = np.random.default_rng(seed=42)

    device = next(model.parameters()).device
    model.eval()
    outputs = [
        torch.zeros(num_regions, device=device, dtype=torch.float) for _ in range(steps)
    ]

    for _ in range(sim_data_length):
        # try:
        noise = 0.1 * rng.standard_normal(size=(steps, num_regions))
        sim_input = torch.vstack(
            [x.to(device) for x in outputs[-steps:]]
        ) + torch.tensor(noise, dtype=torch.float).to(device)
        outputs.append(model(sim_input.unsqueeze(0).to(device))[:, -1, :])
        outputs = [x.to("cpu") for x in outputs]
        # except:
        #     noise = 0.1 * np.random.randn(steps * num_regions)
        #     sim_input = torch.cat(outputs[-steps:]).to(device) + torch.tensor(noise, dtype=torch.float).to(device)
        #     outputs.append(
        #         model(sim_input.unsqueeze(0).to(device)).squeeze(0)
        #     )

    outputs = torch.vstack(outputs)
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=5, neginf=-5)

    try:
        connectome = ConnectivityMeasure(kind="correlation").fit_transform(
            outputs.detach().cpu().numpy()
        )
    except ValueError:
        connectome = np.zeros((outputs.shape[1], outputs.shape[1]))
        return connectome
    return connectome[0]


def get_model_fc_gcn(model, steps, sim_data_length, num_regions, fc, threshold):

    src, des = np.where(np.abs(fc) > threshold)
    edge_idx = np.stack([src, des])
    weights = np.abs(fc[src, des])
    device = next(model.parameters()).device
    outputs = [
        torch.zeros(num_regions, device=device, dtype=torch.float).to(device)
        for _ in range(steps)
    ]

    for _ in range(sim_data_length):
        noise = 0.1 * np.random.randn(steps, num_regions)
        sim_input = torch.vstack(outputs[-steps:]).to(device) + torch.tensor(
            noise, dtype=torch.float
        ).to(device)
        sim_input = Data(
            x=sim_input[-steps:].t(),
            edge_index=torch.tensor(edge_idx, dtype=torch.long).to(device),
            edge_attr=torch.tensor(weights, dtype=torch.float).to(device),
            # y=torch.tensor(bold_data[step], dtype=torch.float).unsqueeze(-1)
        )
        outputs.append(
            model(sim_input.x, sim_input.edge_index, sim_input.edge_attr).t()
        )
    outputs = torch.vstack(outputs)
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=5, neginf=-5)

    try:
        connectome = ConnectivityMeasure(kind="correlation").fit_transform(
            outputs.detach().cpu().numpy()
        )
    except ValueError:
        connectome = np.zeros((outputs.shape[1], outputs.shape[1]))
        return connectome
    return connectome[0]


def evaluate_on_train_end(cfg: DictConfig | ListConfig, test_data, real_fc, model):

    warnings.simplefilter("ignore", FutureWarning)
    work_dir = Path(cfg.work_dir)
    run = work_dir.joinpath(cfg.run_name)
    try:
        df = pd.read_csv(f"{work_dir.name}.csv", index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["r", "p"])

    # Organizing labels into networks
    with open("100p_labels.txt", "r") as f:
        labels = f.readlines()
    labels = [x.removesuffix("\n") for x in labels]
    # neworderlr = np.loadtxt('./neworderlr.txt')
    # neworder = np.array([np.where(neworderlr == i)[0][0] for i in range(len(neworderlr))])

    if run.name in df.index.to_list():
        return

    model.eval()
    model.to(cfg.device)
    model_fc = get_model_fc(model, cfg.data.train.step, 1200, 100)
    torch.cuda.empty_cache()

    recon, recon_err = get_recon(model, test_data, cfg.data.train.step)
    torch.cuda.empty_cache()
    run.joinpath("recon_signal").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recon).to_csv(run.joinpath("recon_signal", "signal.csv"))
    df.at[run.name, "recon_err"] = recon_err

    n_traces = recon.shape[1]
    ncols = 10  # adjust to taste
    nrows = np.ceil(n_traces / ncols).astype(int)  # 18 rows for 360
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.5, nrows * 1.2), sharex=True, sharey=True
    )

    for i, ax in enumerate(axes.flat):
        if i < n_traces:
            ax.plot(
                test_data[:, i],
                color="steelblue",
                linewidth=0.8,
                label="Original",
                rasterized=True,
            )
            ax.plot(
                recon[:, i],
                color="tomato",
                linewidth=0.8,
                label="Reconstruction",
                rasterized=True,
            )
            ax.axis("off")
        else:
            ax.set_visible(False)  # hide unused subplots

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.legend(
        ["Original", "Reconstruction"],
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    run.joinpath("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(run.joinpath("figures", "traces.png"), dpi=150, bbox_inches="tight")
    plt.close()

    r, p = pearsonr(remove_diag(real_fc), remove_diag(model_fc))
    df.at[run.name, "r"] = r
    df.at[run.name, "p"] = p
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(remove_diag(model_fc), remove_diag(real_fc))
    plt.xlim(-0.7, 1.1)
    plt.xticks([-0.5, 0.0, 1.0])
    plt.xlabel("model FC")
    plt.ylim(-0.7, 1.1)
    plt.yticks([-0.5, 0.0, 1.0])
    plt.ylabel("empirical FC")
    plt.text(-0.6, 0.9, "r = {:.2f}, p = {:.0e}".format(r, p))
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(run.joinpath("figures", "correlation_plot.png"))
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # sns.heatmap(real_fc[neworder].T[neworder].T, ax = ax1, vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r', cbar = False, square = True, xticklabels = False, yticklabels = False)
    # sns.heatmap(model_fc[neworder].T[neworder].T, ax = ax2, vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r', cbar = False, square = True, xticklabels = False, yticklabels = False)
    sns.heatmap(
        real_fc,
        ax=ax1,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    sns.heatmap(
        model_fc,
        ax=ax2,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    ax1.set_title("Empirical FC")
    ax2.set_title("Model FC")
    plt.savefig(run.joinpath("figures", "fc_heatmap.png"))
    plt.close()
    run.joinpath("connectomes").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(real_fc, index=labels, columns=labels).to_csv(
        run.joinpath("connectomes", "real.csv")
    )
    pd.DataFrame(model_fc, index=labels, columns=labels).to_csv(
        run.joinpath("connectomes", "model.csv")
    )

    df.index.name = "Scan"
    df = df.sort_index()
    df.to_csv(f"{work_dir.name}.csv")
    print(df["r"].mean(), df["r"].median(), df["r"].std())
    print(df["recon_err"].mean(), df["recon_err"].median(), df["recon_err"].std())


if __name__ == "__main__":
    warnings.simplefilter("ignore", FutureWarning)
    with torch.no_grad():
        work_dir = Path(sys.argv[1])
        try:
            df = pd.read_csv(f"{work_dir.name}.csv", index_col=0)
        except:
            df = pd.DataFrame(columns=["r", "p"])

        # Organizing labels into networks
        with open("100p_labels.txt", "r") as f:
            labels = f.readlines()
        labels = [x.removesuffix("\n") for x in labels]

        for run in tqdm(work_dir.iterdir(), total=len(list(work_dir.iterdir()))):
            if run.name in df.index.to_list():
                continue
            cfg = OmegaConf.load(run.joinpath("config.yaml"))
            model = get_model(cfg.model.name, **cfg.model.kwargs)
            try:
                model.load_state_dict(
                    torch.load(find_best_loss(run))["model_state"], strict=True
                )
            except:
                continue

            cfg.run_name = cfg.run_name.split("preproc")[0]

            train_data, test_data = split_single_subject(
                find_raw_data(Path("data_100p/hcp"), cfg.run_name),
                cfg.data.train_proportion,
            )
            scaler = MinMaxScaler((-1, 1)).fit(train_data)
            test_data = scaler.transform(test_data)
            real_fc = pd.read_csv(
                find_raw_data(Path("data_100p/hcp"), cfg.run_name, suff="connectome"),
                index_col=0,
            ).to_numpy()
            model.eval()
            model.to("cuda:1")
            model_fc = get_model_fc(model, cfg.data.train.step, 1200, 100)
            recon, recon_err = get_recon(model, test_data, cfg.data.train.step)

            n_traces = recon.shape[1]
            ncols = 10  # adjust to taste
            nrows = np.ceil(n_traces / ncols).astype(int)  # 18 rows for 360
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 1.5, nrows * 1.2),
                sharex=True,
                sharey=True,
            )

            for i, ax in enumerate(axes.flat):
                if i < n_traces:
                    ax.plot(
                        test_data[:, i],
                        color="steelblue",
                        linewidth=0.8,
                        label="Original",
                        rasterized=True,
                    )
                    ax.plot(
                        recon[:, i],
                        color="tomato",
                        linewidth=0.8,
                        label="Reconstruction",
                        rasterized=True,
                    )
                    ax.axis("off")
                else:
                    ax.set_visible(False)  # hide unused subplots

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            fig.legend(
                ["Original", "Reconstruction"],
                loc="lower center",
                ncol=2,
                fontsize=8,
                bbox_to_anchor=(0.5, -0.01),
            )
            run.joinpath("figures").mkdir(parents=True, exist_ok=True)
            fig.savefig(
                run.joinpath("figures", "traces.png"), dpi=150, bbox_inches="tight"
            )

            plt.close()

            run.joinpath("recon_signal").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(recon).to_csv(run.joinpath("recon_signal", "signal.csv"))
            df.at[run.name, "recon_err"] = recon_err

            r, p = pearsonr(remove_diag(real_fc), remove_diag(model_fc))
            df.at[run.name, "r"] = r
            df.at[run.name, "p"] = p
            plt.figure(figsize=(4.8, 4.8))
            plt.scatter(remove_diag(model_fc), remove_diag(real_fc))
            plt.xlim(-0.7, 1.1)
            plt.xticks([-0.5, 0.0, 1.0])
            plt.xlabel("model FC")
            plt.ylim(-0.7, 1.1)
            plt.yticks([-0.5, 0.0, 1.0])
            plt.ylabel("empirical FC")
            plt.text(-0.6, 0.9, "r = {:.2f}, p = {:.0e}".format(r, p))
            ax = plt.gca()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            plt.tight_layout()
            run.joinpath("figures", "correlation_plot.png").parent.mkdir(
                parents=True, exist_ok=True
            )
            plt.savefig(run.joinpath("figures", "correlation_plot.png"))
            plt.close()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            sns.heatmap(
                real_fc,
                ax=ax1,
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu_r",
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            sns.heatmap(
                model_fc,
                ax=ax2,
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu_r",
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            ax1.set_title("Empirical FC")
            ax2.set_title("Model FC")
            plt.savefig(run.joinpath("figures", "fc_heatmap.png"))
            plt.close()
            run.joinpath("connectomes").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(real_fc, index=labels, columns=labels).to_csv(
                run.joinpath("connectomes", "real.csv")
            )
            pd.DataFrame(model_fc, index=labels, columns=labels).to_csv(
                run.joinpath("connectomes", "model.csv")
            )

        df.index.name = "Scan"
        df = df.sort_index()
        df.to_csv(f"{work_dir.name}.csv")
        print(df["r"].mean(), df["r"].median(), df["r"].std())
        print(df["recon_err"].mean(), df["recon_err"].median(), df["recon_err"].std())
