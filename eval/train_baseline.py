import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tqdm import tqdm

sys.path.append(".")
from evaluate import evaluate_on_train_end

from models import npi_model_getter
from pytorch_trainer import Trainer  # type: ignore
from utils import (
    BaselineDataset,
    get_loss_fn,
    get_scheduler,
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


class SingleSubjectBrainStateTrainer(Trainer):
    def __init__(self, cfg, model_getter, loss_getter, scheduler_getter):
        super().__init__(
            cfg, model_getter, get_loss_fn=loss_getter, get_scheduler=scheduler_getter
        )
        self.num_steps = cfg.model.kwargs.steps
        self.num_model_steps = cfg.num_epochs * 4

    def model_forward(self, batch):
        """Non-recursive model forward"""
        x, y = batch
        B, N = x.shape
        x = x.reshape(B, self.num_steps, int(N / self.num_steps))
        y = y.reshape(B, -1, self.cfg.data.num_regions)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        return loss, B


def main(cfg):

    if cfg.seed is not None:
        fix_seeds(42)

        # Uncomment below to allow for fine tuning on only testing subjects
    # with open("splits/splits_66p/train.json", "r") as f:
    with open("splits/splits_400p_rnn/train.json", "r") as f:
        data = json.load(f)
    # input_csv_list = [data[sub]["ses-3T"] for sub in data]
    input_csv_list = [data[sub]["ses-1"] for sub in data]
    for inp in tqdm(input_csv_list):
        subject = Path(inp["file_path"])
        tr = inp["tr"]
        cfg.run_name = subject.name.removesuffix("_cleaned-timeseries.csv")
        trainer = SingleSubjectBrainStateTrainer(
            cfg, npi_model_getter, get_loss_fn, get_scheduler
        )

        if len(list(trainer.work_dir.rglob("final_model.pt"))) > 0:
            print("Subject is fininshed, continue...")
            continue

        train_data, test_data = split_single_subject(
            subject, cfg.data.train_proportion, tr, cfg.data.target_tr
        )
        # train_data = train_data[:800]
        # test_data = test_data[:100]

        train_dataset = BaselineDataset(
            train_data,
            cfg.data.train.step,
            noise_strength=cfg.data.noise_strength,
        )
        test_dataset = BaselineDataset(
            test_data,
            cfg.data.val.step,
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
                    None,
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
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
