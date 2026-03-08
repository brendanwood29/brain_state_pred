from torch.utils.data import DataLoader
from .dataset import BrainFuncDataset
from  pathlib import Path

def get_train_loader(train_data_json: str | Path, step: int, batch_size=64, shuffle=True, **kwargs):
    
    return DataLoader(
        BrainFuncDataset(train_data_json, step),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )