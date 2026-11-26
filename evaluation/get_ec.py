import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from models import npi_model_getter
from utils import (
    fix_seeds,
    get_config,
    get_sub_info,
    load_trained_model,
    split_single_subject,
)


def perturb_signal(inp, regions, length, strength, how="impulse"):

    inp = torch.stack(inp[-length:], dim=1)

    if how == "impulse":
        print(inp.shape)
        inp[:, [-1], [regions]] = inp[:, [-1], [regions]] + strength * torch.ones(
            (inp.shape[0], 1, 1), device=inp.device, dtype=inp.dtype
        )  # adding purturbation to signal
        print(inp.shape)

    elif how == "npi":
        strengths = torch.std(inp, dim=1)  # take std on the time dim (1)
        strengths = strengths[:, regions].unsqueeze(1)  # get just the std of the roi
        perturbed_signal = (
            0.5
            * strengths
            * torch.ones((inp.shape[0], 1, 1), device=inp.device, dtype=inp.dtype)
        )  # adding purturbation
        inp[:, [-1], [regions]] = inp[:, [-1], [regions]] + perturbed_signal
        return [inp[:, x, :] for x in range(inp.shape[1])]
    elif how == "hrf":
        hrf = (
            torch.tensor(
                [-0.5, 0, 0.25, 1, 2, 1, 0.25], device=inp.device, dtype=inp.dtype
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        inp[:, :, regions] = inp[:, :, regions] + hrf * torch.ones(
            (inp.shape[0], length, len(regions)), device=inp.device, dtype=inp.dtype
        )  # adding purturbation to signal

    return [inp[:, x, :] for x in range(inp.shape[1])]


def get_ec(model, real_data, steps, pred_len, regions, pert_len, run, device):

    # Create starting points at all possible starting points
    # (steps, regions) len num start pts
    real_inputs = [
        real_data[x : x + steps] for x in range(real_data.shape[0] - steps - pred_len)
    ]

    # array of (num starting points, steps, regions)
    stacked_inputs = torch.stack(real_inputs, dim=0)

    # makes a list of arrays size (num start points, regions) of len steps
    stacked_list = [stacked_inputs[:, x, :] for x in range(stacked_inputs.shape[1])]
    real_inputs = copy.deepcopy(stacked_list)

    # Makes unperturbed prediction
    for _ in range(pred_len):
        # Making input tensor of length steps, stack along time dim
        inter_input_real = torch.stack(real_inputs[-steps:], dim=1).to(device)
        # output is of size (num start points, 1, regions)
        real_inputs.append(model(inter_input_real).squeeze(1).to("cpu"))

    # `real` is time indexed, so slicing [steps:] yeilds just predicted timepoints
    real = torch.stack(real_inputs[steps:], dim=1)
    to_save = torch.stack(real_inputs, dim=1).cpu().numpy()

    real_path = run.joinpath("ecs", "signals", "real")
    real_path.mkdir(parents=True, exist_ok=True)

    # Save (time, regions) array for each starting point
    for i in range(to_save.shape[0]):
        pd.DataFrame(to_save[i]).to_csv(real_path.joinpath(f"start-pt_{i}.csv"))

    pred_path = run.joinpath("ecs", "signals", "pred")
    pred_path.mkdir(parents=True, exist_ok=True)

    ecs = []
    for i, region in enumerate(regions):
        # `pert_inputs` is a time indexed list of size (starting pts, regions)
        pert_inputs = copy.deepcopy(stacked_list)
        pert_inputs[-pert_len:] = perturb_signal(
            inp=pert_inputs,
            regions=region,
            length=pert_len,
            strength=5,
            how="impulse",
        )  # Replace the real signal with perturbed signal

        for _ in range(pred_len):
            inter_input_pert = torch.stack(pert_inputs[-steps:], dim=1).to(device)
            pert_inputs.append(model(inter_input_pert).squeeze(1).to("cpu"))

        pred_path.joinpath(f"pert_{i}").mkdir(parents=True, exist_ok=True)

        # `pert` is time indexed, so slicing [steps:] yeilds just predicted timepoints
        # final size of (starting points, pred_len, regions)
        pert = torch.stack(pert_inputs[steps:], dim=1).cpu()
        to_save = torch.stack(pert_inputs, dim=1).cpu().numpy()

        for j in range(to_save.shape[0]):
            pd.DataFrame(to_save[j]).to_csv(
                pred_path.joinpath(f"pert_{i}", f"start-pt_{j}.csv")
            )
        ecs.append((pert - real).numpy())

    return np.array(ecs)  # perturbed region, starting_point, pred_len, output_regions


def main(cfg: DictConfig | ListConfig):

    # inputs
    label_file = "data/utils/region_labels.txt"
    json_file = "splits/splits_like-npi/train.json"

    with open(label_file, "r") as f:
        labels = f.readlines()
    labels = [x.removesuffix("\n") for x in labels]

    with open(json_file, "r") as f:
        data = json.load(f)

    subject = get_sub_info(cfg.run_name, data)

    tr = subject["tr"]
    data_path = Path(subject["file_path"])
    run = Path(cfg.work_dir).joinpath(cfg.run_name)
    _, test_data = split_single_subject(
        data_path, cfg.data.train_proportion, tr, cfg.data.target_tr
    )

    model = npi_model_getter(cfg.model.name, **cfg.model.kwargs)
    model = load_trained_model(model, run.joinpath("models", "final_model.pt"))
    ecs = get_ec(
        model=model,
        real_data=torch.tensor(test_data),
        steps=cfg.data.train.step,
        pred_len=10,
        regions=[[i] for i in range(len(labels))],
        pert_len=8,
        run=run,
        device=cfg.device,
    )

    ecs = np.permute_dims(
        ecs, (1, 2, 0, 3)
    )  # permute to (starting_point, pred_len, perturbed_region, output_region)
    ecs = ecs.mean(axis=0)
    for i in range(ecs.shape[0]):
        pd.DataFrame(ecs[i], index=labels, columns=labels).to_csv(
            run.joinpath("ecs", "ec_mats", f"ec_{i}.csv")
        )
    ec = ecs.sum(axis=0)
    pd.DataFrame(ec, index=labels, columns=labels).to_csv(run.joinpath("ecs", "ec.csv"))

    ec[np.eye(ec.shape[0]).astype(bool)] = 0
    ec /= ec.max()
    sns.heatmap(
        ec,
        cmap="RdBu_r",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        center=0,
    )
    plt.title("EC")
    plt.tight_layout()
    plt.savefig(run.joinpath("figures", "ec.png"))


if __name__ == "__main__":
    fix_seeds(42)
    cfg = get_config()
    main(cfg)
