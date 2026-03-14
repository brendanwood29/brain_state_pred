from torch.optim import AdamW, Adam
from typing import Iterable

def get_optim(name: str, params: Iterable, lr: float, **kwargs):
    
    configured_optimizers = [
        'adamw',
        'adam'
    ]
    
    if name not in configured_optimizers:
        raise NotImplementedError(f'{name} is not a configured scheduler, `name` must be one of {configured_optimizers}')
    
    if name == 'adamw':
        
        return AdamW(
            params=params,
            lr=lr,
            **kwargs
        )
    elif name == 'adam':
        
        return Adam(
            params=params,
            lr=lr,
            **kwargs
        )