import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
import svgutils.transform as sg
import torch
from dtw import dtw
from matplotlib.colors import TwoSlopeNorm
from nilearn.connectome import ConnectivityMeasure
from scipy.signal import welch
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def get_pearson_avg(arr, axis=None):

    fish = np.tanh(np.clip(arr, a_min=-0.999999, a_max=0.999999))
    mean_fish = np.mean(fish, axis=axis)
    return np.arctanh(mean_fish)


def get_upper_tri(arr):
    return np.triu(arr, k=1).flatten()


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
    b, a = signal.butter(order, (low, high), btype="bandpass")
    return signal.filtfilt(b, a, data, axis=0).copy()


def get_recon_npi_method(model, real_data, steps, device):
    outputs = [real_data[x] for x in range(steps)]
    for i in range(0, real_data.shape[0] - steps, 1):
        sim_input = torch.vstack([x for x in real_data[i : i + steps]]).to(
            device=device, dtype=torch.float
        )
        inter_output = model(sim_input.unsqueeze(0).to(device)).squeeze(0)
        outputs.append(inter_output[-1])
        outputs = [x.to("cpu") for x in outputs]
    outputs = torch.vstack(outputs).cpu().numpy()
    real_data = real_data.cpu().numpy()
    return (
        outputs,
        real_data,
        np.mean(
            (real_data[steps : outputs.shape[0]] - outputs[steps : real_data.shape[0]])
            ** 2
        ),
    )


def get_recon(model, real_data, steps, device):
    outputs = [real_data[x] for x in range(steps)]
    while len(outputs) < real_data.shape[0]:
        sim_input = torch.vstack([x.to(device) for x in outputs[-steps:]]).to(
            device=device, dtype=torch.float
        )
        inter_output = model(sim_input.unsqueeze(0).to(device)).squeeze(0)
        outputs.append(inter_output[-1])
        outputs = [x.to("cpu") for x in outputs]

    outputs = torch.vstack(outputs).cpu().numpy()
    real_data = real_data.cpu().numpy()
    return (
        outputs,
        real_data,
        np.mean(
            (real_data[steps : outputs.shape[0]] - outputs[steps : real_data.shape[0]])
            ** 2
        ),
    )


def get_model_fc_npi_method(model, steps, sim_data_length, num_regions, device):

    rng = np.random.default_rng(seed=42)

    outputs = [
        torch.zeros(num_regions, device=device, dtype=torch.float) for _ in range(steps)
    ]

    for _ in range(sim_data_length):
        noise = 0.1 * rng.standard_normal(size=(steps, num_regions))
        sim_input = torch.vstack(
            [x.to(device) for x in outputs[-steps:]]
        ) + torch.tensor(noise, dtype=torch.float).to(device)
        inter_output = model(sim_input.unsqueeze(0).to(device)).squeeze(0)
        outputs.append(inter_output[-1])
        outputs = [x.to("cpu") for x in outputs]

    outputs = torch.vstack(outputs)
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=5, neginf=-5)

    try:
        connectome = ConnectivityMeasure(kind="correlation").fit_transform(
            outputs.cpu().numpy()
        )
        return connectome[0]
    except ValueError:
        connectome = np.zeros((outputs.shape[1], outputs.shape[1]))
        return connectome


def get_model_fc(pred_signal, steps):

    pred = pred_signal[steps:]
    try:
        connectome = ConnectivityMeasure(kind="correlation").fit_transform(pred)
        return connectome[0]
    except ValueError:
        connectome = np.zeros((pred_signal.shape[1], pred_signal.shape[1]))
        return connectome


def get_signal_r(real_data, recon, steps):
    real_data = real_data[steps:]
    recon = recon[steps:]
    r, _ = pearsonr(real_data, recon, axis=0)
    return r


def get_psd_r(real_data, recon, tr, steps):
    real_data = real_data[steps:]
    recon = recon[steps:]
    real_psd = welch(
        real_data, fs=1 / tr, nperseg=real_data.shape[0], axis=0, return_onesided=True
    )
    recon_psd = welch(
        recon, fs=1 / tr, nperseg=recon.shape[0], axis=0, return_onesided=True
    )
    r, _ = pearsonr(real_psd[1], recon_psd[1], axis=0)
    return r


def get_dtw(real, recon, steps):
    real = real[steps:]
    recon = recon[steps:]

    dtws = []
    norm_dtws = []
    for i in range(real.shape[1]):
        alignment = dtw(recon[:, i], real[:, i])
        dtws.append(alignment.distance)
        norm_dtws.append(alignment.normalizedDistance)

    return np.mean(np.array(dtws)), np.mean(np.array(norm_dtws))


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


def plot_trace_imgs(real, recon, save_path, name):
    real = real.T
    recon = recon.T
    diff = real - recon

    # Match signals colours to real signal
    vmin, vmax = real.min(), real.max()
    if vmin > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmax - vmin) / 2, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Match diff colours to differences between signals
    vmin, vmax = diff.min(), diff.max()
    if vmin > 0:
        diff_norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmax - vmin) / 2, vmax=vmax)
    else:
        diff_norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    panels = [
        ("real signal", real, "viridis", norm),
        ("recon signal", recon, "viridis", norm),
        ("difference (real - recon)", diff, "coolwarm", diff_norm),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)
    for ax, (title, data, cmap, norm) in zip(axes, panels):
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            norm=norm,
        )
        ax.set_title(f"{title}\n({data.shape[0]} regions x {data.shape[1]} timepoints)")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Region")
        fig.colorbar(im, ax=ax, label=name, fraction=0.046, pad=0.04)

    fig.savefig(save_path.joinpath(f"{name}_img.png"), dpi=150)
    plt.close()


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


def plot_fc_diff(real_fc, model_fc, run):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(28, 8),
    )
    ax1, ax2, ax3 = axes

    heatmap_kwargs = dict(
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        center=0,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )

    real_hm = sns.heatmap(real_fc, ax=ax1, **heatmap_kwargs)
    fig.colorbar(real_hm.collections[0], ax=ax1, shrink=0.6, pad=0.04)

    model_hm = sns.heatmap(model_fc, ax=ax2, **heatmap_kwargs)
    fig.colorbar(model_hm.collections[0], ax=ax2, shrink=0.6, pad=0.04)

    diff = model_fc - real_fc
    # Worst the difference could be is +/- 2
    heatmap_kwargs["vmin"] = -2
    heatmap_kwargs["vmax"] = 2
    diff_hm = sns.heatmap(diff, ax=ax3, **heatmap_kwargs)
    fig.colorbar(diff_hm.collections[0], ax=ax3, shrink=0.6, pad=0.04)

    ax1.set_title("Real FC")
    ax2.set_title("Model FC")
    ax3.set_title("Model - Real FC")

    plt.savefig(run.joinpath("figures", "fc_heatmap.png"), bbox_inches="tight")
    plt.close()


def plot_corr(real_fc, model_fc, r, p, run):
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(get_upper_tri(model_fc), get_upper_tri(real_fc))
    plt.xlim(-1.1, 1.1)
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.xlabel("Model FC")
    plt.ylim(-1.1, 1.1)
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.ylabel("Real FC")
    plt.text(-0.9, 0.9, f"r = {r:.2f}, p = {p:.0e}")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(run.joinpath("figures", "correlation_plot.png"))
    plt.close()


def evaluate_on_train_end(
    cfg: DictConfig | ListConfig,
    test_data: torch.Tensor,
    model: torch.nn.Module,
    label_file: str | Path,
    device: str,
):

    work_dir = Path(cfg.work_dir)
    run = work_dir.joinpath(cfg.run_name)
    # Making dirs
    run.joinpath("figures").mkdir(parents=True, exist_ok=True)
    run.joinpath("connectomes").mkdir(parents=True, exist_ok=True)
    run.joinpath("ecs", "ec_mats").mkdir(parents=True, exist_ok=True)
    run.joinpath("recon_signal").mkdir(parents=True, exist_ok=True)

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
    # recon, real_data, recon_err = get_recon_npi_method(
    #     model, test_data, cfg.data.train.step, device, mean, std
    # )
    recon, real_data, recon_err = get_recon(
        model, test_data, cfg.data.train.step, device
    )
    model_fc = get_model_fc_npi_method(
        model, cfg.data.train.step, 1200, len(labels), device
    )
    # model_fc = get_model_fc(recon, cfg.data.train.step)
    real_fc = ConnectivityMeasure(kind="correlation").fit_transform(
        test_data.cpu().numpy()
    )[0]

    plot_trace_imgs(real_data, recon, run.joinpath("figures"), "signal")
    plot_trace_imgs(
        np.abs(np.fft.rfft(real_data, axis=0)),
        np.abs(np.fft.rfft(recon, axis=0)),
        run.joinpath("figures"),
        "fft",
    )
    pd.DataFrame(recon, columns=labels).to_csv(
        run.joinpath("recon_signal", "recon_signal.csv")
    )
    pd.DataFrame(real_data, columns=labels).to_csv(
        run.joinpath("recon_signal", "real_signal.csv")
    )
    df.at[run.name, "recon_err"] = recon_err
    df.at[run.name, "recon_r"] = get_pearson_avg(
        get_signal_r(np.array(test_data), recon, cfg.data.train.step)
    )
    df.at[run.name, "psd_r"] = get_pearson_avg(
        get_psd_r(
            np.array(test_data), recon, tr=cfg.data.target_tr, steps=cfg.data.train.step
        )
    )

    nominal_dtw, norm_dtw = get_dtw(real_data, recon, cfg.data.train.step)
    df.at[run.name, "dtw"] = nominal_dtw
    df.at[run.name, "norm_dtw"] = norm_dtw

    parallel_plotting(
        recon, test_data, run, "all_traces", input_len=cfg.data.train.step
    )
    parallel_plotting(
        np.abs(np.fft.rfft(recon, axis=0)),
        np.abs(np.fft.rfft(test_data, axis=0)),
        run,
        "all_rfft",
    )

    r, p = pearsonr(get_upper_tri(real_fc), get_upper_tri(model_fc))
    df.at[run.name, "r"] = r
    df.at[run.name, "p"] = p

    plot_corr(real_fc, model_fc, r, p, run)

    pd.DataFrame(real_fc, index=labels, columns=labels).to_csv(
        run.joinpath("connectomes", "real.csv")
    )
    pd.DataFrame(model_fc, index=labels, columns=labels).to_csv(
        run.joinpath("connectomes", "model.csv")
    )

    plot_fc_diff(real_fc, model_fc, run)
    df.index.name = "Scan"
    df = df.sort_index()
    df.to_csv(f"{work_dir.joinpath(work_dir.name)}.csv")
    mean_r = get_pearson_avg(df["r"].to_numpy(dtype=np.float64))
    print(
        mean_r,
        df["r"].median(),
        df["r"].apply(lambda x: (x - mean_r) ** 2).sum() / (df.shape[0] - 1),
    )
    print(df["recon_err"].mean(), df["recon_err"].median(), df["recon_err"].std())
