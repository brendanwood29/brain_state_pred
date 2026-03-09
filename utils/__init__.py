from torch.utils.data import DataLoader
from .dataset import BrainFuncDataset
from .loss_functions import get_loss_fn
from .optimizers import get_optim
from .schedulers import get_scheduler
from pathlib import Path

def get_loader(data_json: str | Path, step: int, batch_size=64, shuffle=True, **kwargs):
    
    return DataLoader(
        BrainFuncDataset(data_json, step),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )