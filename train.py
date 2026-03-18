import sys
import torch
import random
import numpy as np
from utils import get_loader, Trainer
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
        x, y = batch
        B, N = x.shape
        x = x.reshape(B, self.num_steps, int(N / self.num_steps))
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss


if __name__ == '__main__':
    
    cfg = get_config()
    if cfg.seed is not None:
        fix_seeds(42)
    
    trainer = BrainStateTrainer(cfg)
    
    train_loader = get_loader(
        data_path=cfg.data.train.data_path,
        step=cfg.data.train.step,
        device=cfg.device
    )
    val_loader = get_loader(
        data_path=cfg.data.val.data_path,
        step=cfg.data.val.step,
        shuffle=cfg.data.val.shuffle,
        device=cfg.device
    )
    
    
    with tqdm(range(cfg.num_epochs)) as pbar:
        for final_model_epochs in pbar:
            trainer.train(train_loader)
            pbar.set_postfix(
                {
                    "train_loss": f"{trainer.loss_epoch[-1]:.4f}", 
                    "val_loss": f"{trainer.last_val_loss:.4f}"
                }, 
                refresh=False
            )
            with torch.no_grad():
                should_stop = trainer.val(val_loader)
            if should_stop:
                print(f'Stopped after {final_model_epochs} epochs due to early stopping.')
                break        
    trainer.training_summary(final_model_epochs, save_final=True)
        
        
        
        

    
