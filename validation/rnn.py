from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ecdf
from tqdm import tqdm

"""

The general idea for the rnn is that 
x(t+dt) = x(t) + change in x(t) + noise

"""


def h(x):
    return np.tanh(x)


def NextTR(W, X_init, noise, dt, down_sample):
    X_res = X_init.copy()
    # only down sampled points e.g. 1 in every 100 points. The simulation happens
    # at a much faster rate, but only return 1 in every X points
    for i in range(down_sample):
        X_res = X_res + (-X_res + np.dot(W.T, h(X_res))) * dt + noise[i]
    return X_res


def RNN(
    W, X_init, length, dt, down_sample, sigma, sub_dir, pert_strength=1.0, pert_step=200
):

    ec_dir = sub_dir.joinpath("ecs")
    ec_dir.mkdir(parents=True, exist_ok=True)

    # num nodes
    n = len(X_init)
    # inital state
    X = [X_init]
    # EC is all zeros to start
    real_EC = np.zeros((n, n))

    pert_count = 0
    for i in range(1, length):
        # noise term at end
        noise = (sigma * np.sqrt(dt)) * np.random.normal(0, 1, (down_sample, n))
        # add next TR
        X.append(NextTR(W, X[-1], noise, dt, down_sample))

        if i % pert_step == 0:
            pert_count += 1

            # for each region, taking the second to last output and get the next output
            # which can be compared to the last output to compute EC
            for j in range(n):
                perted_init = X[-2].copy()
                perted_init[j] += pert_strength
                real_EC[j] += NextTR(W, perted_init, noise, dt, down_sample) - X[-1]
    real_EC /= pert_count
    np.fill_diagonal(real_EC, 0)
    return np.array(X), real_EC


if __name__ == "__main__":
    node_n = 400
    data_length = 1200
    dt = 0.01
    TR = 1
    down_sample = int(TR / dt)
    sigma = 1

    data_dir = Path("data/data_400p/rnn")
    for random_seed in tqdm(range(100), total=100):
        np.random.seed(random_seed)

        sub = random_seed + 1
        sub_dir = data_dir.joinpath(f"sub-{sub}", "ses-1", "func")
        sub_dir.mkdir(parents=True, exist_ok=True)

        # strucutral connections
        SC = np.random.normal(0, 1 / np.sqrt(node_n), (node_n, node_n))

        # no self connections
        np.fill_diagonal(SC, 0)

        # initalize the current BOLD signal
        init = np.random.normal(0, 1, (node_n))

        # do RNN
        signal, real_EC = RNN(
            SC,
            init,
            data_length,
            dt,
            down_sample,
            sigma,
            pert_strength=1.0,
            pert_step=30,
            sub_dir=sub_dir,
        )

        pd.DataFrame(signal).to_csv(
            sub_dir.joinpath(f"sub-{random_seed + 1}_ses-1_cleaned-timeseries.csv")
        )
        pd.DataFrame(real_EC).to_csv(
            sub_dir.joinpath(f"sub-{random_seed + 1}_ses-1_EC.csv")
        )
