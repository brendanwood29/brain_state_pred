from torch import nn

def get_loss_fn(name: str):
    
    configured_schedulers = [
        'mse_loss'
    ]
    
    if name not in configured_schedulers:
        raise NotImplementedError(f'{name} is not a configured loss function, `name` must be one of {configured_schedulers}')
    
    if name == 'mse_loss':
        return nn.MSELoss()