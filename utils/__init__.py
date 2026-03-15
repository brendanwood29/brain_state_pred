from torch.utils.data import DataLoader
from .dataset import BrainFuncDataset, SingleSubjectBrainFuncDataset
from .loss_functions import get_loss_fn
from .optimizers import get_optim
from .schedulers import get_scheduler
from .trainer import Trainer
from pathlib import Path


def get_loader(loader_type: str, data_path: str | Path, step: int, device: str, batch_size=64, shuffle=True, **kwargs):
    
    configured_loaders = [
        'single',
        'all'
    ]
    
    if loader_type not in configured_loaders:
        raise NotImplementedError(f'{loader_type} is not a configured scheduler, `name` must be one of {configured_loaders}')
    
    if loader_type == 'all':
    
        return DataLoader(
            BrainFuncDataset(data_path, step, device),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
    elif loader_type == 'single':
        return DataLoader(
            SingleSubjectBrainFuncDataset(data_path, step, device),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        