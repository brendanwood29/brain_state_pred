import matplotlib.pyplot as plt
import numpy as np

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


def RNN(W, X_init, length, dt, down_sample, sigma, pert_strength=1.0, pert_step=200):

    np.random.seed(0)
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
    node_n = 20
    data_length = 8000

    dt = 0.01
    TR = 1
    down_sample = int(TR / dt)
    sigma = 1

    for random_seed in range(50):
        np.random.seed(random_seed)

        # strucutral connections
        SC = np.random.normal(0, 1 / np.sqrt(node_n), (node_n, node_n))
        # no self connections
        np.fill_diagonal(SC, 0)
        # initalize the current BOLD signal
        init = np.random.normal(0, 1, (node_n))
        # do RNN
        signal, real_EC = RNN(SC, init, data_length, dt, down_sample, sigma)

    # np.savetxt(
    #     "./sim_data/SC_{:03d}nodes_".format(node_n)
    #     + "{:04d}steps_".format(data_length)
    #     + "{:02d}seed.txt".format(random_seed),
    #     SC,
    # )
    # np.savetxt(
    #     "./sim_data/dynamics_{:03d}nodes_".format(node_n)
    #     + "{:04d}steps_".format(data_length)
    #     + "{:02d}seed.txt".format(random_seed),
    #     signal,
    # )
    # np.savetxt(
    #     "./sim_data/real_EC_{:03d}nodes_".format(node_n)
    #     + "{:04d}steps_".format(data_length)
    #     + "{:02d}seed.txt".format(random_seed),
    #     real_EC,
    # )
