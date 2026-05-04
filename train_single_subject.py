import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from evaluate import evaluate_on_train_end
from models import npi_model_getter
from pytorch_trainer import Trainer
from utils import SingleSubjectBrainFuncDataset, get_loss_fn, split_single_subject


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
    def __init__(self, cfg, model_getter, loss_getter):
        super().__init__(cfg, model_getter, get_loss_fn=loss_getter)
        self.num_steps = cfg.model.kwargs.steps

    # def model_forward(self, batch):
    #     """ NPI MLP Model Forward"""
    #     batch = [x.to(self.cfg.device) for x in batch]
    #     x, y = batch
    #     B, N = x.shape
    #     y_hat = self.model(x)
    #     loss = self.loss_fn(y_hat, y)
    #     return loss, B

    def model_forward(self, batch):
        """Transformer Model Forward"""
        batch = [x.to(self.cfg.device) for x in batch]
        x, y = batch
        B, N = x.shape
        x = x.reshape(B, self.num_steps, int(N / self.num_steps))
        y = y.reshape(B, -1, 100)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, B

    # def after_training(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         for data in self.train_loader:
    #             data = [x.to(self.cfg.device) for x in data]
    #             x, y = data
    #             B, N = x.shape
    #             x = x.reshape(B, self.num_steps, int(N / self.num_steps))
    #             # y = y.reshape(B, -1, 100)
    #             y_hat = self.model(x)
    #             print(self.loss_fn(y_hat, y))
    #             pred_signal = reconstruct_signal(
    #                 y_hat,
    #                 self.train_loader.dataset.mean.to(self.cfg.device),
    #                 self.train_loader.dataset.std.to(self.cfg.device),
    #             )
    #             real_signal = reconstruct_signal(
    #                 y.permute(1, 0, 2, 3),
    #                 self.train_loader.dataset.mean.to(self.cfg.device),
    #                 self.train_loader.dataset.std.to(self.cfg.device),
    #             )

    #             for i in range(100):
    #                 plt.figure()
    #                 plt.plot(pred_signal.detach().cpu().numpy()[0, :, i], label="pred")
    #                 plt.plot(real_signal.detach().cpu().numpy()[0, :, i], label="real")
    #                 plt.plot(x.detach().cpu().numpy()[0, :, i], label="input")
    #                 plt.legend()
    #                 plt.savefig(f"test_{i}.png")
    #                 plt.close()

    # def model_forward(self, batch):
    # """GCN Model Forward"""
    #     batch = batch.to(self.cfg.device)
    #     y_hat = self.model(batch.x,  batch.edge_index, batch.edge_attr.unsqueeze(-1), batch.batch)
    #     loss = self.loss_fn(y_hat, batch.y)
    #     return loss, batch.num_graphs

    # def model_forward(self, batch):
    # """STGCN Model Forward"""
    #     batch = [x.to(self.cfg.device) for x in batch]
    #     x, y, idx, weights = batch
    #     y_hat = self.model(x, idx[0, ...], weights[0, ...])
    #     loss = self.loss_fn(y_hat, y)
    #     return loss, x.shape[0]


def reconstruct_signal(coeff, mean, std):
    coeff = (coeff * std.unsqueeze(1)) + mean.unsqueeze(1)
    if not coeff.is_complex():
        coeff = torch.complex(coeff[0], coeff[1])
    signal = torch.fft.irfft(coeff, n=20, dim=1)
    return signal


def main(cfg):

    if cfg.seed is not None:
        fix_seeds(42)

    # input_csv_list = list(Path('data_like-npi/hcp').rglob('**/*timeseries.csv'))

    # Uncomment below to allow for fine tuning on only testing subjects
    with open("splits/test.json", "r") as f:
        data = json.load(f)
    # input_csv_list = [Path(data[sub]['ses-3T']['file_path'].replace('npi', 'julie')) for sub in data]
    input_csv_list = [Path(data[sub]["ses-3T"]["file_path"]) for sub in data]
    input_csv_list.sort()
    for subject in tqdm(input_csv_list):
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
        train_loader = TorchDataLoader(
            SingleSubjectBrainFuncDataset(
                train_data, cfg.data.train.step, strength=cfg.data.strength
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        test_loader = TorchDataLoader(
            SingleSubjectBrainFuncDataset(
                test_data,
                cfg.data.test.step,
                strength=0.0,
                # mean=train_loader.dataset.mean,
                # std=train_loader.dataset.std,
                # scaler=train_loader.dataset.scaler # type: ignore
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        ## single subject stcgn
        # train_loader = TorchDataLoader(
        #     SingleSubjectBrainFuncSTGCNDataset(
        #         train_data,
        #         fc,
        #         cfg.data.train.threshold,
        #         cfg.data.train.step,
        #     ),
        #     batch_size=cfg.batch_size,
        #     shuffle=True
        # )
        # test_loader = TorchDataLoader(
        #     SingleSubjectBrainFuncSTGCNDataset(
        #         test_data,
        #         fc,
        #         cfg.data.train.threshold,
        #         cfg.data.test.step,
        #     ),
        #     batch_size=cfg.batch_size,
        #     shuffle=False
        # )

        # single subject gcn
        # train_loader = PyGDataLoader(
        #     SingleSubjectBrainFuncGCNDataset(
        #         train_data,
        #         fc,
        #         cfg.data.train.threshold,
        #         cfg.data.train.step,
        #     ),
        #     batch_size=cfg.batch_size,
        #     shuffle=True
        # )

        # test_loader = PyGDataLoader(
        #     SingleSubjectBrainFuncGCNDataset(
        #         test_data,
        #         fc,
        #         cfg.data.train.threshold,
        #         cfg.data.test.step,
        #     ),
        #     batch_size=cfg.batch_size,
        #     shuffle=False
        # )

        trainer(train_loader=train_loader, val_loader=test_loader)
        torch.cuda.empty_cache()
        with torch.no_grad():
            evaluate_on_train_end(
                cfg,
                # train_loader.dataset.scaler.transform(test_data), # type: ignore
                test_data,
                fc,
                trainer.model,
            )


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
