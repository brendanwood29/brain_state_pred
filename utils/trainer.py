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
from tqdm import tqdm

class Trainer(ABC):
    
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
        
        if cfg.model.init:
            self.load_model(
                cfg.model.init.weights, 
                weights_only=cfg.model.init.weights_only, 
                strict=cfg.model.init.strict
            )
        
        if cfg.model.freeze_modules:
            self.edit_requires_grad(cfg.model.freeze_modules, requires_grad=False)
        if cfg.model.unfreeze_modules:
            self.edit_requires_grad(cfg.model.unfreeze_modules, requires_grad=True)
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
    
    def __call__(self, train_loader, val_loader):
        
        with tqdm(range(self.cfg.num_epochs)) as pbar:
            for final_model_epochs in pbar:
                self.current_epoch = final_model_epochs
                self.train(train_loader)
                pbar.set_postfix(
                    {
                        "train_loss": f"{self.loss_epoch[-1]:.4f}", 
                        "val_loss": f"{self.last_val_loss:.4f}"
                    }, 
                    refresh=False
                )
                with torch.no_grad():
                    should_stop = self.val(val_loader)
                if should_stop:
                    print(f'Stopped after {self.current_epoch} epochs due to early stopping.')
                    break
        self.training_summary(self.current_epoch, save_final=self.cfg.model.save_last)
    
    
    @abstractmethod
    def model_forward(self, batch):
        """Abstract method for a forward pass of a model

        Args:
            batch: Batch from DataLoader.
        Returns:
            (loss, current_batch_size)
        """
        pass
    
    
    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip.max_norm)
    
    def after_train_batch(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None:
            self.clip_grad_norm()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            self.lr_history.append(self.scheduler.get_last_lr())

    
    def train(self, train_loader: DataLoader):
        loss_iters = 0
        samples_processed = 0
        self.model.train()
        with tqdm(train_loader, disable=not self.cfg.batch_pbar) as pbar:
            for x in pbar:
                pbar.set_description('Training Loop: ')
                loss, batch_size = self.model_forward(x)
                loss_iters += (loss.item() * batch_size)
                samples_processed += batch_size
                pbar.set_postfix({'train_loss_step': f'{loss.item():.4f}'})
                self.step_loss.append(loss.item())
                self.step += 1
                self.after_train_batch(loss)
        self.loss_epoch.append(loss_iters / samples_processed)
        
        
    def val(self, val_loader: DataLoader):
        self.model.eval()
        loss_iters = 0
        samples_processed = 0
        with tqdm(val_loader, disable=not self.cfg.batch_pbar) as pbar:
            pbar.set_description('Validation Loop: ')
            for x in pbar:
                loss, batch_size = self.model_forward(x)
                loss_iters += (loss.item() * batch_size)
                samples_processed += batch_size
                pbar.set_postfix({'val_loss_step': f'{loss.item():.4f}'})
        self.val_loss.append(loss_iters / samples_processed)
        
        if self.last_val_loss < self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.save_after_val()
            
        if self.stopper is not None:
            return self.stopper(self.last_val_loss)
        return False
    
    @property
    def last_val_loss(self):
        return self.val_loss[-1] if self.val_loss else float(torch.inf)
    
    
    def save_after_val(self):
        
        model_name = self.save_model()
        
        self.top_saved_models.append([self.last_val_loss, model_name])
        self.top_saved_models.sort(key=lambda x: x[0])
        if len(self.top_saved_models) > self.num_models_save:
            _, worst_path = self.top_saved_models.pop()
            if worst_path.is_file():
                worst_path.unlink()
            

    def load_model(self, file: str | Path, weights_only: bool, strict: bool = True):
        
        pt_file = torch.load(file)
        
        self.model.load_state_dict(pt_file['model_state'], strict=strict)
        if weights_only:
            return
        self.optimizer.load_state_dict(pt_file['optim_state'])
        self.val_loss.append(pt_file['val_loss'])
        if 'scheduler_state' in pt_file:
            self.scheduler.load_state_dict(pt_file['scheduler_state'])
            
    def save_model(self, model_name: str | None = None):
        out_dir = self.work_dir.joinpath('models')
        out_dir.mkdir(parents=True, exist_ok=True)
        params = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'val_loss': self.last_val_loss,
        }
        if self.scheduler is not None:
            params['scheduler_state'] = self.scheduler.state_dict()
        if model_name is None:
            model_name = f'{self.run_name}-epoch-{self.current_epoch}_best_val_loss_{self.last_val_loss:.4f}.pt'
        model_name = out_dir.joinpath(model_name)
        torch.save(params, model_name)
        
        return model_name

    def edit_requires_grad(self, modules: list, requires_grad: bool, verbose: bool = True):
        params_changed = 0
        if requires_grad:
            action = 'Unfreezing'
        else:
            action = 'Freezing'
        
        for name, param in self.model.named_parameters():
            for keyword in modules:
                if isinstance(keyword, list):
                    # Check if all specify keywords are in param name, if keyword is list
                    if all([word in name for word in keyword]):
                        if verbose:
                            print(f'{action} {name}')
                        param.requires_grad = requires_grad
                        params_changed += torch.numel(param.data)
                
                else:
                    if keyword in name:
                        if verbose:
                            print(f'{action} {name}')
                        param.requires_grad = requires_grad
                        params_changed += torch.numel(param.data)
    
        

    
    def training_summary(self, final_epochs: int, save_final: bool):
        if save_final:
            self.save_model(model_name='final_model.pt')
        
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
        