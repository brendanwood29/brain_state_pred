import sys
import torch
from utils import get_loader, Trainer
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm

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
    
    def model_forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss


if __name__ == '__main__':
    
    cfg = get_config()
    
    trainer = BrainStateTrainer(cfg)
    
    train_loader = get_loader(
        data_json=cfg.data.train.json_file,
        step=cfg.data.train.step,
        device=cfg.device
    )
    val_loader = get_loader(
        data_json=cfg.data.val.json_file,
        step=cfg.data.val.step,
        shuffle=cfg.data.val.shuffle,
        device=cfg.device
    )
    
    
    with tqdm(range(cfg.num_epochs)) as pbar:
        for final_model_epochs in pbar:
            trainer.train(train_loader)
            pbar.set_postfix({"train_loss": f"{trainer.loss_epoch[-1]:.4f}", "val_loss": f"{trainer.last_val_loss:.4f}"}, refresh=False)
            with torch.no_grad():
                should_stop = trainer.val(val_loader)
            if should_stop:
                print(f'Stopped after {final_model_epochs} epochs due to early stopping.')
                break        
    trainer.training_summary(final_model_epochs)
        
        
        
        

    
