import sys
import torch
from utils.callbacks import EarlyStopping
from models import get_model
from utils import get_optim, get_scheduler, get_loss_fn, get_loader
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary
from pathlib import Path


def get_config() -> ListConfig | DictConfig:
    
    default_config_path = "configs/config.yaml"

    cfg = OmegaConf.load(default_config_path)

    if len(sys.argv) > 1:
        user_cfg = OmegaConf.load(sys.argv[1])
        cfg = OmegaConf.merge(cfg, user_cfg)

    return cfg


class Trainer():
    
    def __init__(self, cfg: ListConfig | DictConfig):
        self.model = get_model(
            cfg.model.name, 
            **cfg.model.kwargs 
        )
        self.optimizer = get_optim(
            cfg.optim.name, 
            self.model.parameters(), 
            lr=cfg.optim.lr,
            **cfg.optim.kwargs
        )
        self.scheduler = get_scheduler(
            cfg.scheduler.name, 
            self.optimizer, 
            **cfg.scheduler.kwargs
        )
        self.loss_fn = get_loss_fn(
            cfg.loss.name,
            **cfg.loss.kwargs
        )
        
        if cfg.model.print_summary:
            summary(self.model)
        
        self.scheduler = None
        if cfg.scheduler is not None:
            self.lr_history = []
            self.scheduler = get_scheduler(
                cfg.scheduler.name, 
                self.optimizer, 
                **cfg.scheduler.kwargs
            )
        
        self.stopper = None
        if cfg.callbacks.use_early_stopping:
            self.stopper = EarlyStopping(cfg.callbacks.patience)
        
        self.cfg = cfg
        self.loss_epoch = []
        self.val_loss = []
        self.step_loss = []
        self.run_name = cfg.run_name
        self.work_dir = Path(cfg.work_dir).joinpath(self.run_name)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.best_val_loss = 100
        self.top_saved_models = []
        self.num_models_save = cfg.model.num_models_to_save
        
        if cfg.device is not None:
            self.model.to(cfg.device)
        else:
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def model_forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def after_train_batch(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            self.lr_history.append(self.scheduler.get_last_lr())

    
    def train(self, train_loader: DataLoader):
        loss_iters = 0
        self.model.train()
        for x in train_loader:
            loss = self.model_forward(x)
            loss_iters += loss.item()
            self.step_loss.append(loss.item())
            self.step += 1
            self.after_train_batch(loss)
        self.loss_epoch.append(loss_iters / len(train_loader))
        
        
    def val(self, val_loader: DataLoader):
        self.model.eval()
        loss_iters = 0
        for x in val_loader:
            loss = self.model_forward(x)
            loss_iters += loss.item()
        self.val_loss.append(loss_iters / len(val_loader))
        
        if self.last_val_loss < self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.save_model()
            
        if self.stopper is not None:
            return self.stopper(self.last_val_loss)
        return False
    
    @property
    def last_val_loss(self):
        return self.val_loss[-1] if self.val_loss else 0
    
    
    def save_model(self):
        out_dir = self.work_dir.joinpath('models')
        out_dir.mkdir(parents=True, exist_ok=True)
        params = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'val_loss': self.last_val_loss,
        }
        if self.scheduler is not None:
            params['scheduler_state'] = self.scheduler.state_dict()
        model_name = out_dir.joinpath(f'{self.run_name}_best_val_loss_{self.last_val_loss:.4f}.pt')
        torch.save(params, model_name)
        
        self.top_saved_models.append([self.best_val_loss, model_name])
        self.top_saved_models.sort(key=lambda x: x[0])
        if len(self.top_saved_models) > self.num_models_save:
            _, worst_path = self.top_saved_models.pop()
            if worst_path.is_file():
                worst_path.unlink()
            

    def training_summary(self, final_epochs: int):
        
        fig_dir = self.work_dir.joinpath('figures')
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure()
        plt.plot(range(final_epochs+1), self.loss_epoch, 'r')
        plt.plot(range(final_epochs+1), self.val_loss)
        plt.xlabel('Epoch')
        plt.ylabel(f'{self.cfg.loss.name}')
        plt.legend(['Train Loss', 'Validation Loss'])
        plt.savefig(fig_dir.joinpath('loss_curves.png'))
        
        plt.figure()
        plt.plot(range(self.step), self.lr_history)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.savefig(fig_dir.joinpath('lr.png'))
        
        plt.figure()
        plt.plot(range(self.step), self.step_loss)
        plt.xlabel('Step')
        plt.ylabel(f'{cfg.loss.name}')
        plt.savefig(fig_dir.joinpath('step_loss.png'))
        




if __name__ == '__main__':
    
    cfg = get_config()
    
    trainer = Trainer(cfg)

    train_losses = []
    val_losses = []
    
    train_loader = get_loader(
        data_json='splits/train.json',
        step=3,
        device=cfg.device
    )
    val_loader = get_loader(
        data_json='splits/val.json',
        step=3,
        shuffle=False,
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
        
        
        
        

    
