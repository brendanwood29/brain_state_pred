from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.loader import NeighborLoader
from .dataset import BrainFuncDataset, SingleSubjectBrainFuncDataset, BrainFuncGraphDataset
from .loss_functions import get_loss_fn
from .optimizers import get_optim
from .schedulers import get_scheduler
from .trainer import Trainer
from .make_datasplits import split_single_subject
from pathlib import Path


def get_loader(data_path: str | Path, step: int, device: str, batch_size=64, shuffle=True, **kwargs):
    

    return TorchDataLoader(
        BrainFuncDataset(data_path, step, device),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )
    
    
def get_pyg_loader(data_path: str | Path, threshold: float, step: int, device: str, batch_size=64, shuffle=True, **kwargs):
    
    return PyGDataLoader(
        BrainFuncGraphDataset(data_path, threshold, step, device),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )
    
    