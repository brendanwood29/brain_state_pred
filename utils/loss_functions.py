import torch
from torch import nn


class MSEFCLoss():
    
    def __init__(self, mse_weight: float, fc_weight: float):
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.fc_weight = fc_weight
    
    def __call__(self, y_hat, y):
        
        
        model_fc = self.corcoef(y_hat.mT, y_hat.mT)
        real_fc = self.corcoef(y.mT, y.mT)
        
        return (self.mse_weight * self.mse(y_hat, y)) + (self.fc_weight * self.mse(model_fc, real_fc))
        
    def corcoef(self, x: torch.Tensor, y: torch.Tensor):
        x_bar = x.mean(dim=-1)
        y_bar = y.mean(dim=-1)
    
        x_x_bar = x - x_bar.unsqueeze(-1)
        y_y_bar = y - y_bar.unsqueeze(-1)
        
        num = x_x_bar @ y_y_bar.mT
        den = ((x_x_bar ** 2).sum(dim=-1, keepdim=True) @ (y_y_bar ** 2).sum(dim=-1, keepdim=True).mT) ** 0.5  
        return num / den
        

def get_loss_fn(name: str, **kwargs):
    
    configured_schedulers = [
        'mse_loss',
        'mse_fc_loss'
    ]
    
    if name not in configured_schedulers:
        raise NotImplementedError(f'{name} is not a configured loss function, `name` must be one of {configured_schedulers}')
    
    if name == 'mse_loss':
        return nn.MSELoss()
    elif name == 'mse_fc_loss':
        return MSEFCLoss(
            **kwargs
        )
    