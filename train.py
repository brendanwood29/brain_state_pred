import sys
import torch
import random
import numpy as np
from utils import get_loader, get_pyg_loader, Trainer
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm

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

class BrainStateTrainer(Trainer):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_steps = cfg.model.kwargs.steps
        
    def model_forward(self, batch):
        """ Model forward for the transformer based architecture"""
        batch = [x.to(self.cfg.device) for x in batch]
        x, y = batch
        B, N = x.shape
        x = x.reshape(B, self.num_steps, int(N / self.num_steps))
        y = y.reshape(B, self.num_steps, int(N / self.num_steps))
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, x.shape[0]
# class BrainStateTrainer(Trainer):
    
#     def __init__(self, cfg):
#         super().__init__(cfg)
        
#     def model_forward(self, batch):
        # """ Model forward for the graph based architecture"""
#         batch = batch.to(self.cfg.device)
#         y_hat = self.model(batch.x,  batch.edge_index, batch.edge_attr)
#         loss = self.loss_fn(y_hat, batch.y)
#         return loss, batch.num_graphs

def main(cfg: ListConfig | DictConfig):
    
    if cfg.seed is not None:
        fix_seeds(42)
    
    trainer = BrainStateTrainer(cfg)
    
    train_loader = get_loader(
        data_path=cfg.data.train.data_path,
        step=cfg.data.train.step,
        strength=cfg.data.strength,
        batch_size=cfg.batch_size,
        shuffle=cfg.data.train.shuffle
    )
    val_loader = get_loader(
        data_path=cfg.data.val.data_path,
        step=cfg.data.val.step,
        strength=0.0,
        batch_size=cfg.batch_size,
        shuffle=cfg.data.val.shuffle
    )
    
    # Uncomment below to allow for graph based training
    # train_loader = get_pyg_loader(
    #     data_path=cfg.data.train.data_path,
    #     threshold=cfg.data.train.threshold,
    #     step=cfg.data.train.step,
    # )

    # val_loader = get_pyg_loader(
    #     data_path=cfg.data.val.data_path,
    #     threshold=cfg.data.val.threshold,
    #     step=cfg.data.val.step,
    #     shuffle=cfg.data.val.shuffle,
    # )

    
    trainer(train_loader, val_loader)
        
if __name__ == '__main__':
    
    cfg = get_config()
    main(cfg)

        
        

    
