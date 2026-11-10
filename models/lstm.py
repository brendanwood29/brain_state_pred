import torch
import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        proj_size: int=0,
        bias: bool=True,
        dropout: float=0.0,
        **kwargs
    ):
        super().__init__()

        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            proj_size=proj_size,
            batch_first=True
        )
        self.lin = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x, _ = self.model(x)
        return self.lin(x[:, -1, :])

if __name__ == '__main__':
    
    pass