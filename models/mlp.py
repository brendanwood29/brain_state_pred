import torch.nn as nn
from typing import Tuple, Callable

class MLP(nn.Module):
    
    def __init__(
        self, 
        in_size: int,
        out_size: int, 
        widths: Tuple[int], 
        activation: Callable=nn.GELU
    ) -> None:
        
        super().__init__()
        
        module_list = [
            nn.Linear(in_features=in_size, out_features=widths[0]),
            activation()
        ]
        
        for i in range(1, len(widths) - 1):
            module_list.extend(
                [
                    nn.Linear(in_features=widths[i], out_features=widths[i + 1]),
                    activation()
                ]
            )
        
        module_list.extend(
            [
                nn.Linear(in_features=widths[-1], out_features=out_size)
            ]
        )
        
        self.model = nn.Sequential(*module_list)
    
    
    def forward(self, x):
        
        return self.model(x)
        