import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tqdm import tqdm

from models import npi_model_getter
from pytorch_trainer import Trainer
from utils import (
    SingleSubjectBrainFuncRecursiveDataset,
    evaluate_on_train_end,
    get_loss_fn,
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


def delay_signal(x, prev_x, num_steps, num_steps_back):
    """Combines previous timepoint and current timepoint and
    returns the correct window of the signal"""
    comb = torch.cat([prev_x, x], dim=1)
    delayed_signals = comb[
        :, -num_steps - num_steps_back : -num_steps_back, :
    ]  # Just get the most in the past signal
    return delayed_signals


def get_num_steps_back(num):

    if num < 0.50:
        num_steps_back = 1
    elif num < 0.55:
        num_steps_back = 2
    elif num < 0.60:
        num_steps_back = 3
    elif num < 0.65:
        num_steps_back = 4
    elif num < 0.70:
        num_steps_back = 5
    elif num < 0.75:
        num_steps_back = 6
    elif num < 0.80:
        num_steps_back = 7
    else:
        num_steps_back = 8
    return num_steps_back


class SingleSubjectBrainStateTrainer(Trainer):
    def __init__(self, cfg, model_getter, loss_getter):
        super().__init__(cfg, model_getter, get_loss_fn=loss_getter)
        self.num_steps = cfg.model.kwargs.steps
        self.num_model_steps = cfg.num_epochs * 4

    def model_forward(self, batch):
        """Mamba/transformer/mlp Model Forward"""
        # batch = [x.to(self.cfg.device) for x in batch]
        x, y, prev = batch
        B, N = x.shape
        x = x.reshape(B, self.num_steps, int(N / self.num_steps))
        prev = prev.reshape(B, self.num_steps, int(N / self.num_steps))
        y = y.reshape(B, -1, self.cfg.data.num_regions)
        self.model.eval()
        x = self.schedule_recursion(x, prev)
        self.model.train()
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, B

    @torch.no_grad()
    def schedule_recursion(self, x, prev):
        k_include = self.cfg.k
        prob_include = 1 - (
            k_include / (k_include + torch.e ** (self.step / k_include))
        )
        prob_num = self.step / self.num_model_steps  # Percentage of training finished
        if self.current_mode == "Train":
            if torch.rand(1).item() < prob_include:
                if prob_num < 0.4:
                    return x
                num_steps_back = get_num_steps_back(prob_num)
                x[:, -num_steps_back:, :] = self.get_prior_steps(
                    x, prev, num_steps_back
                )
                return x
            else:
                return x
        else:
            x[:, -self.num_steps :, :] = self.get_prior_steps(x, prev, self.num_steps)
            return x

    def get_prior_steps(self, x, prev, num_steps_back):
        """Recursively builds signal from past signals.
        Allows for the model to learn from its own predictions"""
        delayed_signals = delay_signal(x, prev, self.num_steps, num_steps_back)
        prior_outs = [self.model(delayed_signals)]
        for i in range(1, num_steps_back):
            model_input = torch.cat(
                [
                    delayed_signals,
                    torch.cat(prior_outs, dim=1),
                ],
                dim=1,
            )[:, i:, :]
            prior_outs.append(self.model(model_input))
        return torch.cat(prior_outs, dim=1)


def main(cfg):

    if cfg.seed is not None:
        fix_seeds(42)

    # Uncomment below to allow for fine tuning on only testing subjects
    # with open("splits/splits_400p_aabc/val.json", "r") as f:
    with open("splits/splits_400p/val.json", "r") as f:
        data = json.load(f)
    input_csv_list = [data[sub]["ses-3T"] for sub in data]
    for inp in tqdm(input_csv_list):
        subject = Path(inp["file_path"])
        tr = inp["tr"]
        cfg.run_name = subject.name.removesuffix("_cleaned-timeseries.csv")
        trainer = SingleSubjectBrainStateTrainer(cfg, npi_model_getter, get_loss_fn)

        if len(list(trainer.work_dir.rglob("final_model.pt"))) > 0:
            print("Subject is fininshed, continue...")
            continue

        train_data, test_data = split_single_subject(subject, cfg.data.train_proportion)
        fc = pd.read_csv(
            subject.with_name(subject.name.replace("cleaned-timeseries", "connectome")),
            index_col=0,
        ).to_numpy()

        # single subject brain func
        train_dataset = SingleSubjectBrainFuncRecursiveDataset(
            train_data,
            cfg.data.train.step,
            noise_strength=cfg.data.noise_strength,
            target_tr=cfg.data.target_tr,
            tr=tr,
        )
        test_dataset = SingleSubjectBrainFuncRecursiveDataset(
            test_data,
            cfg.data.val.step,
            target_tr=cfg.data.target_tr,
            tr=tr,
            noise_strength=0.0,
        )

        if dist.is_initialized():
            dist.barrier()
        trainer(train_dataset=train_dataset, val_dataset=test_dataset)
        torch.cuda.empty_cache()
        if trainer.trainer_rank == 0:
            with torch.no_grad():
                evaluate_on_train_end(
                    cfg,
                    # train_dataset.scaler.transform(test_data), # type: ignore
                    test_dataset.data,
                    fc,
                    trainer.model,
                    label_file="data/utils/400p_labels.txt",
                    # mean=train_dataset.mean,
                    # std=train_dataset.std,
                    mean=torch.tensor(0),
                    std=torch.tensor(1),
                    device=trainer.trainer_device,
                )
        if dist.is_initialized():
            dist.barrier()

        print(f"{cfg.run_name}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
