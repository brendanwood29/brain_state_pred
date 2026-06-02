import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from models import npi_model_getter
from pytorch_trainer import Trainer
from utils import (
    SingleSubjectBrainFuncDataset,
    SingleSubjectBrainFuncRecursiveDataset,
    SingleSubjectFFTFuncDataset,
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
    comb = torch.cat([prev_x, x], dim=1)
    delayed_signals = [
        comb[:, -num_steps - step : -step, :] for step in range(1, num_steps_back + 1)
    ]
    delayed_signals = list(
        reversed(delayed_signals)
    )  # Reverse so list is oldest to newest
    return delayed_signals


class SingleSubjectBrainStateTrainer(Trainer):
    def __init__(self, cfg, model_getter, loss_getter):
        super().__init__(cfg, model_getter, get_loss_fn=loss_getter)
        self.num_steps = cfg.model.kwargs.steps

    # def model_forward(self, batch):
    #     """NPI MLP Model Forward"""
    #     batch = [x.to(self.cfg.device) for x in batch]
    #     x, y = batch
    #     B, N = x.shape
    #     x = x.reshape(B, self.num_steps, int(N / self.num_steps))
    #     y = y.reshape(B, -1, self.cfg.data.num_regions)
    #     y_hat = self.model(x)
    #     loss = self.loss_fn(y_hat, y[:, 0, :])
    #     return loss, B

    # def model_forward(self, batch):
    #     """Transformer Model Forward"""
    #     batch = [x.to(self.cfg.device) for x in batch]
    #     x, y, prev = batch
    #     B, N = x.shape
    #     x = x.reshape(B, self.num_steps, int(N / self.num_steps))
    #     prev = prev.reshape(B, self.num_steps, int(N / self.num_steps))
    #     y = y.reshape(B, -1, self.cfg.data.num_regions)
    #     x = self.schedule_recursion(x, prev)
    #     y_hat = self.model(x)
    #     loss = self.loss_fn(y_hat, y)
    #     return loss, B

    def model_forward(self, batch):
        """Mamba Model Forward"""
        batch = [x.to(self.cfg.device) for x in batch]
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
        k_include = 20
        k_num = 50
        prob_include = 1 - (
            k_include / (k_include + torch.e ** (self.step / k_include))
        )
        prob_num = 1 - (k_num / (k_num + torch.e ** (self.step / k_num)))

        if self.current_mode == "Train":
            if torch.rand(1).item() < prob_include:
                if 0.2 <= prob_num < 0.3:
                    num_steps_back = 1
                elif 0.3 <= prob_num < 0.4:
                    num_steps_back = 2
                elif 0.4 <= prob_num < 0.5:
                    num_steps_back = 3
                elif 0.5 <= prob_num < 0.6:
                    num_steps_back = 4
                elif 0.6 <= prob_num < 0.7:
                    num_steps_back = 5
                elif 0.7 <= prob_num < 0.8:
                    num_steps_back = 6
                elif 0.8 <= prob_num < 0.9:
                    num_steps_back = 7
                elif 0.9 <= prob_num:
                    num_steps_back = 8
                else:
                    return x
                print(f"\nincluding {num_steps_back}")
                b, t, r = x.shape
                delayed_signals = delay_signal(x, prev, self.num_steps, num_steps_back)
                delayed_signals = torch.stack(delayed_signals, dim=0).reshape(-1, t, r)
                prev_outs = self.model(delayed_signals).reshape(-1, b, 1, r)
                prev_outs = torch.cat(
                    # [self.model(inp) for inp in delayed_signals],
                    [out for out in prev_outs],
                    dim=1,
                )
                x[:, -num_steps_back:, :] = prev_outs
                return x

            else:
                return x
        else:
            b, t, r = x.shape
            delayed_signals = delay_signal(x, prev, self.num_steps, self.num_steps)
            # prev_outs = torch.cat([self.model(inp) for inp in delayed_signals], dim=1)
            delayed_signals = torch.stack(delayed_signals, dim=0).reshape(-1, t, r)
            prev_outs = self.model(delayed_signals).reshape(-1, b, 1, r)
            prev_outs = torch.cat(
                [out for out in prev_outs],
                dim=1,
            )
            x[:, -self.num_steps :, :] = prev_outs
            return x

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


# def reconstruct_signal(coeff, mean, std):
#     coeff = (coeff * std.unsqueeze(1)) + mean.unsqueeze(1)
#     if not coeff.is_complex():
#         coeff = torch.complex(coeff[0], coeff[1])
#     signal = torch.fft.irfft(coeff, n=20, dim=1)
#     return signal


def main(cfg):

    if cfg.seed is not None:
        fix_seeds(42)

    # input_csv_list = list(Path('data_like-npi/hcp').rglob('**/*timeseries.csv'))

    # Uncomment below to allow for fine tuning on only testing subjects
    with open("splits_1000p/val.json", "r") as f:
        data = json.load(f)
    input_csv_list = [data[sub]["ses-3T"] for sub in data]
    # input_csv_list = [
    #     Path(data[sub]["ses-3T"]["file_path"].replace("data_100p", "data_like-npi"))
    #     for sub in data
    # ]
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
        # for predicting fourier
        # train_dataset = SingleSubjectFFTFuncDataset(
        #     train_data,
        #     cfg.data.train.step,
        #     noise_strength=cfg.data.noise_strength,
        #     pred_len=cfg.data.pred_len,
        #     target_tr=cfg.data.target_tr,
        #     tr=tr,
        # )
        # train_loader = TorchDataLoader(
        #     train_dataset,
        #     batch_size=cfg.batch_size,
        #     shuffle=True,
        # )
        # test_dataset = SingleSubjectFFTFuncDataset(
        #     test_data,
        #     cfg.data.test.step,
        #     pred_len=cfg.data.pred_len,
        #     target_tr=cfg.data.target_tr,
        #     tr=tr,
        #     noise_strength=0.0,
        #     mean=train_loader.dataset.mean,
        #     std=train_loader.dataset.std,
        # )
        # test_loader = TorchDataLoader(
        #     test_dataset,
        #     batch_size=cfg.batch_size,
        #     shuffle=False,
        # )

        # single subject brain func
        train_dataset = SingleSubjectBrainFuncRecursiveDataset(
            train_data,
            cfg.data.train.step,
            noise_strength=cfg.data.noise_strength,
            target_tr=cfg.data.target_tr,
            tr=tr,
        )
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        test_dataset = SingleSubjectBrainFuncRecursiveDataset(
            test_data,
            cfg.data.test.step,
            target_tr=cfg.data.target_tr,
            tr=tr,
            noise_strength=0.0,
        )
        test_loader = TorchDataLoader(
            test_dataset,
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
                # train_dataset.scaler.transform(test_data), # type: ignore
                train_dataset.data,
                fc,
                trainer.model,
                label_file="1000p_labels.txt",
                # mean=train_dataset.mean,
                # std=train_dataset.std,
                mean=torch.tensor(0),
                std=torch.tensor(1),
            )


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
