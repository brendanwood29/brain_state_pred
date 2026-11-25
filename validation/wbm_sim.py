import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.io import loadmat
from scipy.stats.mstats import winsorize

# Globals
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_data = loadmat("WBM_simulation_data/Human_66.mat")
C = np.array(load_data["C"])
anat_labels = load_data["anat_lbls"]
order = np.array(load_data["Order"][0]) - 1
C = (C[order].T)[order].T
anat_labels = anat_labels[order]
with open("WBM_simulation_data/66p_labels.txt", "w") as f:
    for label in anat_labels:
        f.write(label + "\n")

C = C  # structural connectivity
N = C.shape[0]  # number of nodes
w = 0.55  # self-excitation scaling factor
G = 3.5  # long-range scaling factor
JN = 0.2609  # overall excitatory strength
I0 = 0.3255  # mean of background input
tau0 = 0.002  # noise time constant
sigma = 0.005  # noise amplitude
tauS = 0.1  # synaptic time constant
gamma = 0.641  # saturation factor for gating variable
dt = 0.002  # dt of Newton's method


def F(I, a=270.0, b=108.0, d=0.154):
    "Transfer function used to determine population-firing rate r with synaptic input I"

    return (a * I - b) / (1.0 - np.exp(-d * (a * I - b)))


def fbold(t, p=2, taubold=1.25, o=2.25):
    "Boynton gamma function used as a HRF kernel"

    kernel = np.zeros_like(t)
    idx = t >= o
    kernel[idx] = (
        ((t[idx] - o) / taubold) ** (p - 1)
        / math.factorial(p - 1)
        * np.exp(-(t[idx] - o) / taubold)
    )

    return kernel


def data_simulation(T, S_start, Ib_start, Ib_in, start, step):

    range_t = np.arange(0, T, dt)
    nt = range_t.size

    S = np.zeros((nt, N))
    S[0] = S_start
    r = np.zeros((nt, N))
    B = torch.zeros(nt, N).type(torch.float32).to(device)
    IB = np.zeros((nt, N))

    fbold_vec = torch.from_numpy(fbold(range_t)[: int(60 / dt)]).type(torch.float32)
    reversed_fbold_vec = fbold_vec.flip(0).to(device)

    I = np.zeros(N)
    Ib = Ib_start
    IB[0] = Ib

    for idx in range(1, nt):
        I = w * JN * S[idx - 1] + G * JN * C @ S[idx - 1] + Ib
        r[idx] = F(I)
        S[idx] = S[idx - 1] + dt * (
            gamma * r[idx] * (1 - S[idx - 1]) - S[idx - 1] / tauS
        )
        Ib = Ib + (dt / tau0) * (-(Ib - I0) + np.sqrt(tau0) * sigma * Ib_in[idx])
        IB[idx] = Ib

    S_res = S[start::step]
    Ib_res = IB[start::step]
    S = torch.from_numpy(S).type(torch.float32).to(device)

    for idx in range(N):
        padding_input = torch.nn.functional.pad(
            S[:, idx], (fbold_vec.size(0) - 1, 0), mode="constant", value=0
        )
        B[:, idx] = torch.nn.functional.conv1d(
            padding_input.view(1, 1, -1), reversed_fbold_vec.view(1, 1, -1)
        )[0][0]

    B_res = B[start::step].detach().cpu().numpy()

    del S
    del B

    return S_res, Ib_res, B_res


def main():

    for random_seed in range(50):
        TR = 0.72
        step = int(TR / dt)
        start = step - 1
        T = int(TR * 1200)
        S_start = np.zeros(N)
        r_start = np.zeros(N)
        np.random.seed(random_seed)
        Ib_in = np.random.randn(int(T / dt), N)
        Ib_start = sigma * Ib_in[0]
        S_res, Ib_res, B_res = data_simulation(T, S_start, Ib_start, Ib_in, start, step)
        np.savetxt(
            "./sim_data/dynamics_raw_T_"
            + str(T)
            + "_TR_"
            + str(int(1000 * TR))
            + "_seed_"
            + "{:03d}".format(random_seed)
            + ".txt",
            B_res,
        )

        pert_step = 500
        pert_T = 5
        pert_start_list = list(
            range(2000, B_res.shape[0] - int(pert_T / TR), pert_step)
        )
        realEC = np.zeros((N, N))
        for pert_start in pert_start_list:
            S_start = S_res[pert_start - 1]
            Ib_start = Ib_res[pert_start - 1]
            Ib_in_unpert = Ib_in[
                start + pert_start * step : start
                + pert_start * step
                + int(pert_T / dt),
                :,
            ]
            _, _, B_unpert = data_simulation(
                pert_T, S_start, Ib_start, Ib_in_unpert, start, step
            )
            unpert = B_unpert[4]
            for i in range(N):
                Ib_in_pert = Ib_in_unpert.copy()
                perturb = np.zeros(N)
                perturb[i] = 5.0
                Ib_in_pert[: int(0.1 / dt)] = Ib_in_pert[: int(0.1 / dt)] + perturb
                _, _, B_pert = data_simulation(
                    pert_T, S_start, Ib_start, Ib_in_pert, start, step
                )
                realEC[i] = realEC[i] + (B_pert[4] - unpert)
        realEC = realEC / len(pert_start_list)
        np.savetxt(
            "./sim_data/real_EC_seed_" + "{:03d}".format(random_seed) + ".txt", realEC
        )


def bandpass(data, low, high, fs, order=4):
    ny = fs / 2
    low = low / ny
    high = high / ny
    b, a = signal.butter(order, (low, high), btype="bandpass")  # type: ignore
    return signal.filtfilt(b, a, data, axis=0).copy()


def organize_data():
    data_path = Path(
        "WBM_simulation_data/sim_data/dynamics_raw_T_8640_TR_720_seed_000.txt"
    )
    ec = Path("WBM_simulation_data/sim_data/real_EC_seed_000.txt")

    data = np.loadtxt(data_path)
    ec = np.loadtxt(ec)

    save_path = Path("data/data_66p/wbm_sim/sub-01/ses-1/func")
    save_path.mkdir(parents=True, exist_ok=True)

    data = data[21:]
    data = winsorize(data, limits=[0.05, 0.05], axis=0)
    data = bandpass(data, 0.01, 0.1, 1 / 0.72)

    data = pd.DataFrame(data)
    cols = data.columns
    for col in cols:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    plt.figure()
    plt.plot(data[0][:80])
    plt.savefig("test.png")

    data.to_csv(save_path.joinpath("sub-01_ses-1_cleaned-timeseries.csv"))
    ec = pd.DataFrame(ec)
    ec.to_csv(save_path.joinpath("sub-01_ses-1_EC.csv"))


def plot_data(data):
    plt.figure()
    plt.plot(data[0].to_numpy()[20:])
    plt.savefig("test.png")


if __name__ == "__main__":
    main()
    # organize_data()
