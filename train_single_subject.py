import json
import sys
import torch
import random
import numpy as np
import pandas as pd
from models import npi_model_getter
from pytorch_trainer import Trainer
from torch.utils.data import DataLoader as TorchDataLoader
from utils import SingleSubjectBrainFuncDataset, split_single_subject
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path


def fix_seeds(seed=42):
    print(f'Fixing random seed to {seed}')
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
    
    def __init__(self, cfg, model_getter):
        super().__init__(cfg, model_getter)
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
        """ Transformer Model Forward """
        batch = [x.to(self.cfg.device) for x in batch]
        x, y = batch
        B, N = x.shape
        x = x.reshape(B, self.num_steps, int(N / self.num_steps))
        # y = y.reshape(B, self.num_steps, int(N / self.num_steps))
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, B

    
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


def main(cfg):
    
    if cfg.seed is not None:
        fix_seeds(42)
    
    # input_csv_list = list(Path('data_like-npi/hcp').rglob('**/*timeseries.csv'))

    # Uncomment below to allow for fine tuning on only testing subjects
    with open('splits/test.json', 'r') as f:
        data = json.load(f)
    input_csv_list = [Path(data[sub]['ses-3T']['file_path']) for sub in data]
    
    for subject in tqdm(input_csv_list):
        cfg.run_name = subject.name.removesuffix('_cleaned-timeseries.csv')
        trainer = SingleSubjectBrainStateTrainer(cfg, npi_model_getter)
        
        if len(list(trainer.work_dir.rglob('final_model.pt'))) > 0:
            print('Subject is fininshed, continue...')
            continue
        
        train_data, test_data = split_single_subject(subject, cfg.data.train_proportion)
        fc = pd.read_csv(subject.with_name(subject.name.replace('cleaned-timeseries', 'connectome')), index_col=0).to_numpy()
        # single subject brain func
        train_loader = TorchDataLoader(
            SingleSubjectBrainFuncDataset(
                train_data,
                cfg.data.train.step,
                strength=cfg.data.strength
            ),
            batch_size=cfg.batch_size,
            shuffle=True
        )
        test_loader = TorchDataLoader(
            SingleSubjectBrainFuncDataset(
                test_data,
                cfg.data.test.step,
                strength=0.0
            ),
            batch_size=cfg.batch_size,
            shuffle=False
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
            
            
if __name__ == '__main__':
    
    cfg = get_config()
    main(cfg)