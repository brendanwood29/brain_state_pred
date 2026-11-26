import random
import sys
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def get_sub_info(run_name: str, json_file: dict[str, dict[str, str | float]]):
    parts = run_name.split("_")
    return json_file[parts[0]][parts[1]]


def load_trained_model(model: nn.Module, weights: str | Path):
    pt_file = torch.load(weights)
    weights = OrderedDict()  # type: ignore
    for k, v in pt_file["model_state"].items():
        weights[k.removeprefix("module.")] = v
    model.load_state_dict(weights)
    model.eval()

    return model


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
