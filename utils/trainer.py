import torch
from utils.callbacks import EarlyStopping
from models import get_model
from utils import get_optim, get_scheduler, get_loss_fn
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
import matplotlib.pyplot as plt
from torchinfo import summary
from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import OmegaConf


class Trainer(ABC):
    
    def __init__(self, cfg: ListConfig | DictConfig):
        self.model = get_model(
            cfg.model.name, 
            **cfg.model.kwargs 
        )
        if cfg.model.init is not None:
            self.model.load_state_dict(
                torch.load(cfg.model.init.weights),
                strict=cfg.model.init.strict
            )
        self.optimizer = get_optim(
            cfg.optim.name, 
            self.model.parameters(), 
            lr=cfg.optim.lr,
            **cfg.optim.kwargs
        )
        self.loss_fn = get_loss_fn(
            cfg.loss.name,
            **cfg.loss.kwargs
        )
        self.scheduler = None
        if cfg.scheduler is not None:
            self.lr_history = []
            self.scheduler = get_scheduler(
                cfg.scheduler.name, 
                self.optimizer, 
                **cfg.scheduler.kwargs
            )
        
        if cfg.model.print_summary:
            summary(self.model)
        
        
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
        self.best_val_loss = torch.inf
        self.top_saved_models = []
        self.num_models_save = cfg.model.num_models_to_save
        
        if cfg.device is not None:
            self.model.to(cfg.device)
        else:
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            
        OmegaConf.save(
            self.cfg,
            self.work_dir.joinpath('config.yaml')
        )
    
    @abstractmethod
    def model_forward(self, batch):
        pass
    
    def after_train_batch(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            self.lr_history.append(self.scheduler.get_last_lr())

    
    def train(self, train_loader: DataLoader):
        loss_iters = 0
        samples_processed = 0
        self.model.train()
        for x in train_loader:
            loss = self.model_forward(x)
            loss_iters += (loss.item() * x[0].shape[0])
            samples_processed += x[0].shape[0]
            self.step_loss.append(loss.item())
            self.step += 1
            self.after_train_batch(loss)
        self.loss_epoch.append(loss_iters / samples_processed)
        
        
    def val(self, val_loader: DataLoader):
        self.model.eval()
        loss_iters = 0
        samples_processed = 0
        for x in val_loader:
            loss = self.model_forward(x)
            loss_iters += (loss.item() * x[0].shape[0])
            samples_processed += x[0].shape[0]
        self.val_loss.append(loss_iters / samples_processed)
        
        if self.last_val_loss < self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.save_model()
            
        if self.stopper is not None:
            return self.stopper(self.last_val_loss)
        return False
    
    @property
    def last_val_loss(self):
        return self.val_loss[-1] if self.val_loss else float(torch.inf)
    
    
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
            

    def training_summary(self, final_epochs: int, save_final: bool):
        if save_final:
            out_dir = self.work_dir.joinpath('models')
            out_dir.mkdir(parents=True, exist_ok=True)
            params = {
                'model_state': self.model.state_dict(),
                'optim_state': self.optimizer.state_dict(),
                'val_loss': self.last_val_loss,
            }
            model_name = out_dir.joinpath('final_model.pt')
            torch.save(params, model_name)
        
        fig_dir = self.work_dir.joinpath('figures')
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'Model finished training with best validation {self.cfg.loss.name}: {self.best_val_loss:.4f}')
        
        plt.figure()
        plt.plot(range(final_epochs+1), self.loss_epoch, 'r')
        plt.plot(range(final_epochs+1), self.val_loss)
        plt.xlabel('Epoch')
        plt.ylabel(f'{self.cfg.loss.name}')
        plt.legend(['Train Loss', 'Validation Loss'])
        plt.tight_layout()
        plt.savefig(fig_dir.joinpath('loss_curves.png'))
        plt.close()
        
        if self.scheduler is not None:
            plt.figure()
            plt.plot(range(self.step), self.lr_history)
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.tight_layout()
            plt.savefig(fig_dir.joinpath('lr.png'))
            plt.close()
        
        plt.figure()
        plt.plot(range(self.step), self.step_loss)
        plt.xlabel('Step')
        plt.ylabel(f'{self.cfg.loss.name}')
        plt.tight_layout()
        plt.savefig(fig_dir.joinpath('step_loss.png'))
        plt.close()
        