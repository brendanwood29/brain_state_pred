from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer

def get_scheduler(name: str, optim: Optimizer,**kwargs):
    
    configured_schedulers = [
        'cosine'
    ]
    
    if name not in configured_schedulers:
        raise NotImplementedError(f'{name} is not a configured scheduler, `name` must be one of {configured_schedulers}')
    
    if name == 'cosine':
        
        scheduler = OneCycleLR(
            optimizer=optim,
            **kwargs
        )
    
    return scheduler

