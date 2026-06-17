import shutil
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
import svgutils.transform as sg
import torch
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch_geometric.data import Data


def remove_diag(arr):
    return arr[~np.eye(arr.shape[0], dtype=bool)]


def sort_fn(path: Path):
    loss = path.name.split("_")[-1].removesuffix(".pt")
    return float(loss)


def find_best_loss(path, which_model):
    if which_model == "last":
        model = path.joinpath("models/final_model.pt")
        return model
    else:
        models = [x for x in path.rglob("**/*best_val_loss*.pt")]
        return sorted(models, key=sort_fn)[0]


def find_raw_data(path, name, suff="timeseries"):
    return list(path.rglob(f"**/{name}*{suff}.csv"))[0]


def bandpass(data, low, high, fs, order=4):
    ny = fs / 2
    low = low / ny
    high = high / ny
    b, a = signal.butter(order, high, btype="lowpass")  # type: ignore
    return signal.filtfilt(b, a, data, axis=0).copy()


def get_recon(model, real_data, steps, device, mean, std):
    model.eval()
    mean = mean.to(device)
    std = std.to(device)
    outputs = [torch.tensor(real_data[x]) for x in range(steps)]
    real_data = torch.tensor(real_data)
    # for i in range(0, real_data.shape[0] - steps, 1):
    while len(outputs) < real_data.shape[0]:
        sim_input = torch.vstack([x.to(device) for x in outputs[-steps:]]).to(
            device=device, dtype=torch.float
        )
        # sim_input = torch.vstack([x for x in real_data[i : i + steps]]).to(
        #     device=device, dtype=torch.float
        # )
        # inter_output = model(sim_input.unsqueeze(0).to(device)).reshape(2, 5, -1)
        inter_output = model(sim_input.unsqueeze(0).to(device)).squeeze(0)
        # inter_output = (inter_output * std) + mean
        # inter_output = torch.fft.irfft(
        #     torch.complex(inter_output[0], inter_output[1]), dim=0
        # )
        outputs.append(inter_output[-1])
        # outputs.extend([x for x in inter_output])
        # outputs.append(model(sim_input.unsqueeze(0).to(device)).squeeze(0)[-1, :])
        # outputs.extend([x for x in model(sim_input.unsqueeze(0).to(device)).squeeze(0)])
        outputs = [x.to("cpu") for x in outputs]
    outputs = torch.vstack(outputs).detach().cpu().numpy()
    # outputs = bandpass(outputs, 0.01, 0.1, 0.5)
    real_data = real_data.cpu().numpy()
    return outputs, np.mean(
        (real_data[steps : outputs.shape[0]] - outputs[steps : real_data.shape[0]]) ** 2
    )


def get_model_fc(model, steps, sim_data_length, num_regions, device, mean, std):

    rng = np.random.default_rng(seed=42)

    mean = mean.to(device)
    std = std.to(device)
    model.eval()
    outputs = [
        torch.zeros(num_regions, device=device, dtype=torch.float) for _ in range(steps)
    ]

    for _ in range(sim_data_length):
        noise = 0.1 * rng.standard_normal(size=(steps, num_regions))
        sim_input = torch.vstack(
            [x.to(device) for x in outputs[-steps:]]
        ) + torch.tensor(noise, dtype=torch.float).to(device)
        # outputs.append(model(sim_input.unsqueeze(0).to(device)).squeeze(0)[-1])
        inter_output = model(sim_input.unsqueeze(0).to(device)).squeeze(0)
        # inter_output = model(sim_input.unsqueeze(0).to(device)).reshape(2, 5, -1)
        # inter_output = (inter_output * std) + mean
        # inter_output = torch.fft.irfft(
        #     torch.complex(inter_output[0], inter_output[1]), dim=0
        # )
        outputs.append(inter_output[-1])
        # outputs.extend([x for x in inter_output])
        outputs = [x.to("cpu") for x in outputs]

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


def get_model_fc_gcn(model, steps, sim_data_length, num_regions, fc, threshold, device):

    src, des = np.where(np.abs(fc) > threshold)
    edge_idx = np.stack([src, des])
    weights = np.abs(fc[src, des])
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


def plot_traces(i, test_data, recon, save_path, input_len=None):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(
        test_data,
        color="steelblue",
        linewidth=0.8,
        label="Original",
    )
    ax.plot(
        recon,
        color="tomato",
        linewidth=0.8,
        label="Reconstruction",
    )
    if input_len is not None:
        ax.axvline(x=7, color="r", ls="dashed")
    fig.tight_layout(pad=0.5)
    fig.savefig(save_path.joinpath(f"panel_{i}.svg"))
    plt.close(fig)


def parallel_plotting(recon, test_data, run, fname, input_len=None):

    n_traces = recon.shape[1]
    ncols = np.ceil(np.sqrt(n_traces)).astype(int)
    nrows = np.ceil(n_traces / ncols).astype(int)
    panel_path = run.joinpath("figures", "panels")
    panel_path.mkdir(parents=True, exist_ok=True)
    Parallel(-1, backend="loky")(
        delayed(plot_traces)(
            i, test_data[:, i], recon[:, i], panel_path, input_len=input_len
        )
        for i in range(n_traces)
    )
    panel_w, panel_h = 600, 200  # must match figsize * dpi (default 100)

    fig = sg.SVGFigure()
    fig.set_size((f"{ncols * panel_w}px", f"{nrows * panel_h}px"))

    panels = sorted(panel_path.glob("panel_*.svg"))
    elements = []
    for idx, path in enumerate(panels):
        row, col = divmod(idx, ncols)
        svg = sg.fromfile(str(path))
        root = svg.getroot()
        root.moveto(col * panel_w, row * panel_h)
        elements.append(root)

    fig.append(elements)
    fig.save(run.joinpath("figures", f"{fname}.svg"))
    shutil.rmtree(panel_path)


def evaluate_on_train_end(
    cfg: DictConfig | ListConfig,
    test_data,
    real_fc,
    model,
    label_file,
    mean,
    std,
    device,
):

    warnings.simplefilter("ignore", FutureWarning)
    work_dir = Path(cfg.work_dir)
    run = work_dir.joinpath(cfg.run_name)
    try:
        df = pd.read_csv(f"{work_dir.joinpath(work_dir.name)}.csv", index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["r", "p"])

    # Organizing labels into networks
    with open(label_file, "r") as f:
        labels = f.readlines()
    labels = [x.removesuffix("\n") for x in labels]

    if run.name in df.index.to_list():
        return

    model.eval()
    model.to(device)
    model_fc = get_model_fc(
        model, cfg.data.train.step, 1200, len(labels), device, mean, std
    )
    torch.cuda.empty_cache()

    recon, recon_err = get_recon(
        model, test_data, cfg.data.train.step, device, mean, std
    )
    torch.cuda.empty_cache()
    run.joinpath("recon_signal").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recon).to_csv(run.joinpath("recon_signal", "signal.csv"))
    df.at[run.name, "recon_err"] = recon_err

    parallel_plotting(
        recon, test_data, run, "all_traces", input_len=cfg.data.train.step
    )
    parallel_plotting(
        np.abs(np.fft.rfft(recon, axis=0)),
        np.abs(np.fft.rfft(test_data, axis=0)),
        run,
        "all_rfft",
    )
    r, p = pearsonr(remove_diag(real_fc), remove_diag(model_fc))
    df.at[run.name, "r"] = r  # type: ignore
    df.at[run.name, "p"] = p  # type: ignore
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
    df.to_csv(f"{work_dir.joinpath(work_dir.name)}.csv")
    print(df["r"].mean(), df["r"].median(), df["r"].std())
    print(df["recon_err"].mean(), df["recon_err"].median(), df["recon_err"].std())
