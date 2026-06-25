import json
import random
import sys
from pathlib import Path
from typing import OrderedDict

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from models import npi_model_getter
from utils import (
    SingleSubjectBrainFuncRecursiveDataset,
    evaluate_on_train_end,
    split_single_subject,
)


def fix_seeds(seed=42):
    print(f"Fixing random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config() -> ListConfig | DictConfig:

    default_config_path = "configs/default_config.yaml"

    cfg = OmegaConf.load(default_config_path)

    if len(sys.argv) > 1:
        user_cfg = OmegaConf.load(sys.argv[1])
        cfg = OmegaConf.merge(cfg, user_cfg)

    return cfg


if __name__ == "__main__":
    cfg = get_config()
    subject = Path(
        "data/data_400p/hcp/sub-107220/ses-3T/func/sub-107220_ses-3T_task-rest_acq-lr_space-MNIICBM152_desc-preproc_cleaned-timeseries.csv"
    )

    with open("splits/splits_400p/val.json", "r") as f:
        data = json.load(f)
    input_csv_list = [data[sub]["ses-3T"] for sub in data]
    for subject in input_csv_list:
        tr = subject["tr"]
        subject = Path(subject["file_path"])
        cfg.run_name = subject.name.removesuffix("_cleaned-timeseries.csv")
        train_data, test_data = split_single_subject(subject, 0.8)

        fc = pd.read_csv(
            subject.with_name(subject.name.replace("cleaned-timeseries", "connectome")),
            index_col=0,
        ).to_numpy()
        train_dataset = SingleSubjectBrainFuncRecursiveDataset(
            train_data,
            cfg.data.train.step,
            noise_strength=cfg.data.noise_strength,
            target_tr=cfg.data.target_tr,
            tr=tr,
        )
        test_dataset = SingleSubjectBrainFuncRecursiveDataset(
            test_data,
            cfg.data.train.step,
            noise_strength=cfg.data.noise_strength,
            target_tr=cfg.data.target_tr,
            tr=tr,
        )
        model = npi_model_getter(cfg.model.name, **cfg.model.kwargs)
        pt_file = torch.load(
            f"results/mamba_val_set/{cfg.run_name}/models/final_model.pt"
        )

        weights = OrderedDict()
        for k, v in pt_file["model_state"].items():
            weights[k.removeprefix("module.")] = v
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            evaluate_on_train_end(
                cfg,
                test_dataset.data,
                fc,
                model,
                label_file="data/utils/400p_labels.txt",
                mean=torch.tensor(0),
                std=torch.tensor(1),
                device="cuda:0",
            )
